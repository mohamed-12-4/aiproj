"""Pinned official Rein++/Mask2Former runtime, isolated from the DPT suite."""
from pathlib import Path
import json
import os
import time
import uuid
import io

import modal

LOCAL = Path(__file__).resolve().parent
AUTHOR = LOCAL / "third_party/rein_plus_author/projects/rein++"
app = modal.App("coralscapes-v2-reinpp-mask2former")
volume = modal.Volume.from_name("coral-training")
leases = modal.Dict.from_name("coralscapes-v2-reinpp-owners",create_if_missing=True)
if modal.is_local():
    image = (modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10")
             .apt_install("libgl1", "libglib2.0-0", "libgomp1")
             .pip_install("numpy==1.26.4", "setuptools<81", "wheel", "packaging")
             .pip_install("torch==2.1.0", "torchvision==0.16.0",
                          extra_options="--index-url https://download.pytorch.org/whl/cu118")
             .pip_install("mmcv==2.1.0", "mmengine==0.10.7", "mmsegmentation==1.2.2", "mmdet==3.3.0",
                          "timm==0.9.16", "Pillow==10.4.0", "scipy==1.13.1", "einops==0.8.0",
                          "ftfy==6.3.1", "regex==2024.11.6", "cityscapesscripts==2.3.0",
                          "safetensors==0.6.2", "PyYAML==6.0.3", "opencv-python==4.10.0.84",
                          extra_options="-f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html --only-binary=mmcv")
             .env({"PYTHONPATH":"/app", "PYTHONUNBUFFERED":"1", "MPLCONFIGDIR":"/tmp/matplotlib",
                   "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "OMP_NUM_THREADS":"4"})
             .add_local_dir(AUTHOR / "reinpp", "/app/reinpp", ignore=["**/__pycache__/**"])
             .add_local_dir(AUTHOR / "configs", "/app/author_configs")
             .add_local_file(LOCAL / "third_party/rein_plus_author/rein/optimizers/peft_optimizer_constructor.py",
                             "/app/author_optimizer.py")
             .add_local_file(LOCAL / "third_party/rein_plus_author/configs/_base_/models/rein_dinov2_mask2former.py",
                             "/app/author_train_head.py")
             .add_local_dir(LOCAL / "reinpp_training", "/app/reinpp_training", ignore=["**/__pycache__/**"])
             .add_local_dir(LOCAL / "coralscapes_metadata", "/app/coralscapes_metadata"))
else:
    image = modal.Image.debian_slim()


@app.function(image=image, cpu=(1,1), memory=(4096,4096), timeout=600, retries=0)
def environment_remote():
    import torch, torchvision, mmcv, mmengine, mmseg, mmdet
    from mmcv.ops import MultiScaleDeformableAttention
    import reinpp
    reinpp.register_modules()
    return {"torch":torch.__version__, "torchvision":torchvision.__version__, "mmcv":mmcv.__version__,
            "mmengine":mmengine.__version__, "mmseg":mmseg.__version__, "mmdet":mmdet.__version__,
            "reinpp":"official source 38b688ea7236b0aa58d1186dcaf37a240816bd81"}


@app.function(image=image, cpu=(2,2), memory=(4096,4096), timeout=1800, retries=0,
              volumes={"/workspace":volume})
def prepare_remote():
    import hashlib
    os.environ["HF_HUB_OFFLINE"]="0"
    from huggingface_hub import HfApi,hf_hub_download
    volume.reload()
    # Original Meta bytes, authenticated by the SHA published by the Rein++
    # authors. The public replica is only a transport source, never a fine-tuned initialization.
    repo="PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m"
    filename="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    digest="8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
    target=Path("/workspace/pretrained/reinpp")/filename
    ready=json.loads(Path("/workspace/data/coralscapes_v2_paper_8a4567b2fbec/ready.json").read_text())
    if ready["counts"]!={"train":1790,"val":170,"test":473}:
        raise RuntimeError("Native V2 data is incomplete")
    if not target.exists():
        info=HfApi(token=False).model_info(repo)
        print(f"Fetching original Meta backbone from checksum-verified replica {repo}@{info.sha}",flush=True)
        downloaded=hf_hub_download(repo,filename,revision=info.sha,token=False,local_dir=target.parent)
    h=hashlib.sha256()
    with target.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    if h.hexdigest()!=digest:
        raise RuntimeError("The downloaded base checkpoint differs from the authors' required original Meta SHA-256")
    metadata={"path":str(target),"sha256":digest,"original_meta_checkpoint":True,"replica_transport_repo":repo,
              "data_revision":ready["revision"]}
    (target.parent/"verified.json").write_text(json.dumps(metadata,indent=2)+"\n")
    volume.commit()
    return metadata


def claim(key):
    owner=modal.current_function_call_id()
    if not leases.put(key,owner,skip_if_exists=True) and leases.get(key)!=owner:
        raise RuntimeError("Another function call already owns this run; refusing a duplicate")


@app.function(image=image,gpu="L40S",cpu=(4,4),memory=(24576,24576),timeout=900,
              retries=0,max_containers=1,scaledown_window=2,single_use_containers=True,
              volumes={"/workspace":volume})
def probe_remote(group: str,deadline: float,attempt: int = 1):
    claim("probe:"+group+":"+str(attempt))
    if time.time()>=deadline:raise RuntimeError("Probe deadline expired")
    volume.reload()
    from reinpp_training.runner import probe
    from reinpp_training.config import write_json
    failure=Path("/workspace/checkpoints")/group/"probe_failure.json"
    if failure.exists():
        failure.replace(failure.with_name(f"probe_failure_attempt_{attempt-1}.json"))
    try:
        result=probe(group);volume.commit()
        return {k:v for k,v in result.items() if k!="optimizer_groups"}
    except BaseException as error:
        write_json(Path("/workspace/checkpoints")/group/"probe_failure.json",
                   dict(error_type=type(error).__name__,message=str(error)[:2000]))
        volume.commit();raise


@app.function(image=image,gpu="L40S",cpu=(4,4),memory=(24576,24576),timeout=22500,
              retries=0,max_containers=1,scaledown_window=2,single_use_containers=True,
              volumes={"/workspace":volume})
def train_remote(group: str,run_id: str,deadline: float):
    claim("train:"+group)
    if time.time()>=deadline-600:raise RuntimeError("The original training budget deadline expired")
    volume.reload()
    from reinpp_training.runner import run
    try:return run(group,run_id,deadline,commit=volume.commit)
    finally:volume.commit()


@app.function(image=image,cpu=(.125,.125),memory=(256,256),timeout=24000,retries=0)
def wait_remote(call_id: str):
    """Passive keeper; no GPU submission, restart or continuation logic."""
    return modal.FunctionCall.from_id(call_id).get()


@app.local_entrypoint()
def main(environment_only: bool = False, prepare_only: bool = False, probe_only: bool = False,
         group: str = "reinpp-v2-mhgelu-m2f-s42-40e-20261101", training_budget_usd: float = 14.5,
         retry_probe: bool = False):
    if os.environ.get("MODAL_PROFILE") != "mohamed-elashmony05":
        raise ValueError("The new Rein++ experiment belongs to mohamed-elashmony05")
    if prepare_only:
        result=prepare_remote.remote()
        (LOCAL/"analysis/paper_v2_sources/reinpp_original_backbone.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    elif environment_only:
        result = environment_remote.remote()
        output = LOCAL / "analysis/paper_v2_sources/reinpp_environment.json"
        output.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    else:
        import re
        from reinpp_training.config import write_json
        if not re.fullmatch(r"[A-Za-z0-9_-]+",group):raise ValueError("Invalid run group")
        if not 0<training_budget_usd<=14.5:raise ValueError("The authorized remaining budget permits at most $14.50 for training")
        path=LOCAL/"analysis/paper_v2_sources/mohamed-elashmony05"/(group+"-submission.json")
        state=json.loads(path.read_text()) if path.exists() else dict(group=group,profile="mohamed-elashmony05",
                    run_id=uuid.uuid4().hex,epochs=40,precision="BF16 backbone / FP32 Mask2Former",gpu="L40S",
                    multi_head=True,gelu=True,lr=1e-4,training_budget_usd=training_budget_usd)
        if state.get("train_call_id"):
            print(f"Already submitted training: {state['train_call_id']}");return
        if retry_probe and state.get("probe_call_id"):
            old=modal.FunctionCall.from_id(state["probe_call_id"])
            graph=old.get_call_graph()
            if not graph or graph[0].status.name not in ("FAILURE","TERMINATED","TIMEOUT"):
                raise RuntimeError("Only a confirmed terminated/failed probe can be explicitly retried")
            state.setdefault("previous_probe_calls",[]).append(state.pop("probe_call_id"))
            state["probe_attempt"]=state.get("probe_attempt",1)+1
            write_json(path,state)
        if not state.get("probe_call_id"):
            prepared=prepare_remote.remote()
            state["base_checkpoint"]=prepared
            call=probe_remote.spawn(group,time.time()+900,state.get("probe_attempt",1))
            state.update(probe_call_id=call.object_id,probe_dashboard=app.get_dashboard_url())
            write_json(path,state)
            if probe_only:
                keeper=wait_remote.spawn(call.object_id)
                state["probe_keeper_call_id"]=keeper.object_id;write_json(path,state)
                print(f"Submitted full paper-architecture GPU probe: {call.object_id}")
                print(f"Dashboard: {app.get_dashboard_url()}");return
        call=modal.FunctionCall.from_id(state["probe_call_id"])
        call.get(timeout=1000)
        b=io.BytesIO();volume.read_file_into_fileobj(f"/checkpoints/{group}/probe.json",b)
        result=json.loads(b.getvalue())
        write_json(LOCAL/"analysis/paper_v2_sources"/(group+"-probe.json"),result)
        if result["status"]!="passed":raise RuntimeError("The architecture/EMA GPU probe did not pass")
        if probe_only:
            print(json.dumps({k:v for k,v in result.items() if k!="optimizer_groups"},indent=2));return
        # Includes a margin for native-image loading, two full validation sweeps
        # (raw+EMA), final raw+EMA testing, checkpoint writes and startup.
        seconds=17880*result["seconds_per_step"]*1.2 + (170*4+473*2)*result["validation_seconds_per_image"] + 700
        hourly=2.331648
        estimate=seconds/3600*hourly
        state.update(estimated_seconds=seconds,estimated_training_cost_usd=estimate,
                     probe_step_seconds=result["seconds_per_step"],probe_eval_seconds_per_image=result["validation_seconds_per_image"])
        write_json(path,state)
        if estimate>training_budget_usd:
            raise RuntimeError(f"The prepared 40-epoch job is estimated at ${estimate:.2f}, above its ${training_budget_usd:.2f} remaining-credit allocation; no training submitted")
        deadline=time.time()+training_budget_usd/hourly*3600
        call=train_remote.spawn(group,state["run_id"],deadline)
        state.update(train_call_id=call.object_id,train_deadline_utc_timestamp=deadline,train_dashboard=app.get_dashboard_url())
        write_json(path,state)
        keeper=wait_remote.spawn(call.object_id)
        state["train_keeper_call_id"]=keeper.object_id;write_json(path,state)
        print(f"Submitted fresh Rein++ multi-head/GELU + Mask2Former for 40 epochs: {call.object_id}")
        print(f"Estimated cost: ${estimate:.2f}; training runtime cap: ${training_budget_usd:.2f}")
        print(f"Manifest: {path}");print(f"Dashboard: {app.get_dashboard_url()}")
