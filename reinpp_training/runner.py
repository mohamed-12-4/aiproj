"""Forty fresh epochs with official Mask2Former loss and effective-token EMA."""
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .config import experiment_config, write_json
from .data import build_dataset, train_loader
from .ema import EffectiveTokenEMA
from .metrics import finite, mapped_metrics, summarize
from .model import BASE_SHA, adapted_state, build_model, build_optimizer, selected_weights


def atomic_save(value,path):
    path=Path(path);temporary=path.with_suffix(".tmp")
    torch.save(value,temporary);temporary.replace(path)


def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def random_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),
                cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_random(state):
    random.setstate(state["python"]);n=state["numpy"]
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state["cpu"]);torch.cuda.set_rng_state_all(state["cuda"])


def check_gradient_paths(model):
    reins=model.backbone.reins
    norms={}
    for name in ("learnable_tokens_a","learnable_tokens_b"):
        grad=getattr(reins,name).grad
        if grad is None:raise RuntimeError(f"No gradient for {name}")
        value=grad.flatten(1).norm(dim=1)
        if not torch.isfinite(value).all() or not (value>0).all():
            raise RuntimeError(f"A Rein++ layer has a zero/nonfinite {name} gradient")
        norms[name]=value.detach().cpu().tolist()
    frozen=[n for n,p in model.backbone.named_parameters() if not n.startswith("reins.") and p.grad is not None]
    if frozen:raise RuntimeError(f"Frozen pretrained parameters received gradients: {frozen[:5]}")
    if model.decode_head.cls_embed.weight.grad is None:
        raise RuntimeError("Mask2Former classifier has no gradient")
    return norms


@torch.no_grad()
def evaluate(model,split="val",limit=None):
    model.eval();dataset=build_dataset(split)
    counts=torch.zeros((96,96),dtype=torch.long,device="cuda");by_site={}
    started=time.time()
    for i in range(min(len(dataset),limit) if limit else len(dataset)):
        data=dataset[i]
        prediction=model.test_step(dict(inputs=[data["inputs"]],data_samples=[data["data_samples"]]))[0]
        target=prediction.gt_sem_seg.data.squeeze(0).to("cuda",dtype=torch.long)
        predicted=prediction.pred_sem_seg.data.squeeze(0).to("cuda",dtype=torch.long)
        if predicted.shape!=target.shape or predicted.min()<0 or predicted.max()>95:
            raise RuntimeError("Native prediction geometry or semantic IDs are invalid")
        valid=target!=0
        image_counts=torch.bincount(target[valid]*96+predicted[valid],minlength=96*96).reshape(96,96)
        counts+=image_counts
        site=Path(dataset.get_data_info(i)["img_path"]).parent.name
        if site not in by_site:by_site[site]=torch.zeros_like(counts)
        by_site[site]+=image_counts
        del prediction,target,predicted,data,image_counts
        if (i+1)%50==0:print(f"EVAL {split}: {i+1}/{len(dataset)}",flush=True)
    cpu=counts.cpu();native=summarize(cpu)
    result=dict(native=native,miou=native["miou"],health_4=mapped_metrics(cpu),
                by_site={site:mapped_metrics(value.cpu()) for site,value in by_site.items()},
                inference="native 1024x2048; sliding 512x512, stride 341",seconds=time.time()-started,
                images=min(len(dataset),limit) if limit else len(dataset))
    return finite(result)


def probe(group):
    seed_all();model=build_model();model.train()
    optimizer=build_optimizer(model);ema=EffectiveTokenEMA(model)
    gradient_report={}
    optimizer.optimizer.register_step_pre_hook(lambda *_:gradient_report.update(check_gradient_paths(model)))
    optimizer.optimizer.register_step_post_hook(lambda *_:ema.update(model))
    dataset=build_dataset("train")
    from mmengine.dataset import pseudo_collate
    batch=pseudo_collate([dataset[i] for i in range(4)])
    # Verify that native void is ignored while named Background remains supervised.
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample
    sample=SegDataSample(metainfo=dict(img_shape=(16,16),ori_shape=(16,16)))
    mask=torch.zeros((1,16,16),device="cuda",dtype=torch.long);mask[:,8:]=14
    sample.gt_sem_seg=PixelData(data=mask)
    instances,_=model.decode_head._seg_data_to_instance_data([sample])
    if instances[0].labels.tolist()!=[14]:raise RuntimeError("Void/Background supervision is incorrect")
    # Source-native readout audit, before feature-pyramid resampling.
    captures={};original=model.backbone.reins.forward
    def record(inp,out,layer,**kwargs):
        result=original(inp,out,layer,**kwargs)
        if layer in model.backbone.out_indices:captures[layer]=result.detach()
        return result
    model.backbone.reins.forward=record
    try:
        with torch.no_grad():
            processed=model.data_preprocessor(batch,training=True)
            features=model.backbone(processed["inputs"][:1])["feature"]
            errors=[]
            for feature,index,ratio in zip(features,model.backbone.out_indices,(4,2,1,.5)):
                expected=captures[index].permute(0,2,1).reshape(1,1024,32,32).contiguous()
                if ratio!=1:expected=torch.nn.functional.interpolate(expected,scale_factor=ratio,mode="bilinear",align_corners=False)
                difference=float((feature-expected).abs().max());errors.append(difference)
                if difference!=0:raise RuntimeError("A native Rein++ readout missed its adapter output")
        del captures,processed,features,expected
    finally:model.backbone.reins.forward=original
    times=[];torch.cuda.reset_peak_memory_stats()
    for step in range(6):
        torch.cuda.synchronize();started=time.perf_counter()
        metrics=model.train_step(batch,optimizer)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        if not all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()):
            raise RuntimeError("Non-finite Mask2Former training loss")
        if step>=2:times.append(elapsed)
        print(f"REINPP PROBE step={step} seconds={elapsed:.4f} loss={float(metrics['loss']):.5f}",flush=True)
    raw=evaluate(model,"val",limit=2)
    with ema.apply(model):averaged=evaluate(model,"val",limit=2)
    # The inference export must retain the full EMA token matrix, which can
    # have rank higher than 16. Verify exported weights reproduce live EMA.
    model.eval()
    def small_prediction():
        with torch.no_grad():
            x=torch.zeros((1,3,64,64),device="cuda")
            classes,masks=model.decode_head(model.backbone(x))
            return classes[-1].detach().clone(),masks[-1].detach().clone()
    with ema.apply(model):expected=small_prediction()
    exported=ema.inference_state(model)
    with selected_weights(model,dict(model=exported)):actual=small_prediction()
    export_errors=[float((a-b).abs().max()) for a,b in zip(actual,expected)]
    if any(export_errors):raise RuntimeError("The dense-token EMA inference export differs from live EMA")
    del exported,actual,expected
    groups=[];names={id(p):n for n,p in model.named_parameters()}
    for group_item in optimizer.optimizer.param_groups:
        groups.append(dict(names=[names[id(p)] for p in group_item["params"]],lr=group_item["lr"],weight_decay=group_item["weight_decay"]))
    tokens=[g for g in groups if any("learnable_tokens" in n or "reins.scale" in n for n in g["names"])]
    if not tokens or any(g["weight_decay"]!=0 or g["lr"]!=1e-4 for g in tokens):
        raise RuntimeError("The published optimizer exemptions were not applied")
    import statistics
    result=dict(status="passed",gpu=torch.cuda.get_device_name(),physical_batch=4,input_hw=[512,512],
                seconds_per_step=statistics.median(times),timed_steps=times,
                validation_seconds_per_image=(raw["seconds"]+averaged["seconds"])/4,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                gradient_norms=gradient_report,rein_readout_max_abs=errors,
                background_target_id=14,ignored_target_id=0,ema_updates=ema.num_updates,
                ema_export_roundtrip_max_abs=export_errors,
                optimizer_groups=groups,precision="BF16 backbone; FP32 Mask2Former",base_sha256=BASE_SHA)
    out=Path("/workspace/checkpoints")/group
    write_json(out/"probe.json",result)
    return result


def run(group,run_id,deadline,commit=lambda:None):
    config=experiment_config();out=Path("/workspace/checkpoints")/group;out.mkdir(parents=True,exist_ok=True)
    if (out/"completed.json").exists():return json.loads((out/"completed.json").read_text())
    seed_all();model=build_model();optim=build_optimizer(model);ema=EffectiveTokenEMA(model)
    start,updates,best,next_validation=0,0,{"raw":-1.,"ema":-1.},10000
    state=torch.load(out/"last.pt",map_location="cpu",weights_only=True) if (out/"last.pt").exists() else None
    if state:
        if state["kind"]!="reinpp_raw_training_resume" or state["run_id"]!=run_id or state["config"]!=config:
            raise ValueError("This checkpoint does not belong to this new Rein++ experiment")
        model.load_state_dict(state["model"],strict=False);optim.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"]);start,updates,best=state["epoch"]+1,state["updates"],state["best"]
        next_validation=state["next_validation"];restore_random(state["rng"])
    del state
    write_json(out/"config.json",dict(config,run_id=run_id,original_backbone_sha256=BASE_SHA))
    optim.optimizer.register_step_post_hook(lambda *_:ema.update(model))
    dataset=build_dataset("train");completed=start;previous_epoch=None
    try:
        for epoch in range(start,40):
            if time.time()+max(300,previous_epoch or 300)+600>=deadline:
                write_json(out/"budget_stopped.json",dict(epoch_completed=completed,updates=updates));commit()
                return dict(status="budget_stopped",epoch_completed=completed)
            loader=train_loader(dataset,epoch);model.train();started=time.time();loss_sum=0.
            print(f"EPOCH {epoch+1}/40 Rein++ multi-head GELU / Mask2Former; updates={updates}",flush=True)
            for index,batch in enumerate(loader):
                lr=1e-4*max(0.,1-updates/40000)**.9
                for parameter_group in optim.optimizer.param_groups:parameter_group["lr"]=lr
                logs=model.train_step(batch,optim)
                loss=float(logs["loss"]);loss_sum+=loss;updates+=1
                if not __import__('math').isfinite(loss):raise RuntimeError("Nonfinite Mask2Former loss")
                if (index+1)%50==0:print(f"STEP {index+1}/{len(loader)} loss={loss:.5f} lr={lr:.8f}",flush=True)
            row=dict(epoch=epoch,epoch_completed=epoch+1,optimizer_updates=updates,train_loss=loss_sum/len(loader),lr=lr)
            if updates>=next_validation or epoch==39:
                raw=evaluate(model)
                with ema.apply(model):averaged=evaluate(model)
                row.update(raw=raw,ema=averaged)
                for name,metrics in (("raw",raw),("ema",averaged)):
                    if metrics["miou"]>best[name]:
                        best[name]=metrics["miou"]
                        weights=adapted_state(model) if name=="raw" else ema.inference_state(model)
                        atomic_save(dict(model=weights,epoch=epoch,metrics=metrics,config=config,
                                         kind="reinpp_selected_"+name),out/f"best_{name}.pt")
                        write_json(out/f"best_{name}_metrics.json",dict(metrics,epoch=epoch));del weights
                while next_validation<=updates:next_validation+=10000
                print(f"VALIDATION epoch={epoch+1}: raw={raw['miou']:.6f}, EMA={averaged['miou']:.6f}",flush=True)
            atomic_save(dict(kind="reinpp_raw_training_resume",epoch=epoch,updates=updates,run_id=run_id,
                             model=adapted_state(model),optimizer=optim.state_dict(),ema=ema.state_dict(),
                             rng=random_state(),best=best,next_validation=next_validation,config=config),out/"last.pt")
            completed=epoch+1;previous_epoch=time.time()-started
            row["seconds_with_checkpoint"]=previous_epoch
            with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(finite(row))+"\n")
            write_json(out/"progress.json",dict(epoch_completed=completed,optimizer_updates=updates,
                        stop_after_epochs=40,last_epoch_seconds=previous_epoch))
            commit()
    except BaseException as error:
        write_json(out/"failure.json",dict(error_type=type(error).__name__,message=str(error)[:2000],epoch_completed=completed));commit();raise
    result=dict(status="completed_40_epochs",epoch_completed=40,optimizer_updates=updates,best_validation=best)
    del optim,ema,dataset,loader;gc.collect();torch.cuda.empty_cache()
    probe_result=json.loads((out/"probe.json").read_text())
    test_estimate=473*probe_result["validation_seconds_per_image"]*1.3+120
    if time.time()+test_estimate<deadline:
        results={}
        for name in ("raw","ema"):
            if time.time()+test_estimate>=deadline:
                result["test_evaluation_pending"]=True
                break
            checkpoint=torch.load(out/f"best_{name}.pt",map_location="cpu",weights_only=True)
            with selected_weights(model,checkpoint):results[name]=evaluate(model,"test")
            results[name]["selected_epoch"]=checkpoint["epoch"]+1
            write_json(out/"test_results.json",results);commit();del checkpoint
        result["test"]=results
    else:result["test_evaluation_pending"]=True
    write_json(out/"completed.json",result);commit();return result
