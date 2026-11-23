"""Cross-check converted cached backbone/reference ops against prior Modal outputs."""
import json
import torch

from .assets import ROOT,OUT,write_json
from .runtime import build_rein,rein_outputs
from .probe import tensor


def main():
    old_root=ROOT/"analysis/external_eval/neom_pilot"
    old=json.loads((old_root/"reinpp_results.json").read_text())
    fetched=json.loads((OUT/"rein_checkpoint.json").read_text())
    if old["checkpoint_sha256"]!=fetched["sha256"]:raise RuntimeError("Reference used a different checkpoint")
    from external_pilot_runtime import read_pilot
    _,rows=read_pilot(old_root/"data")
    expected={r["point_id"]:r["predicted_native"] for r in old["predictions"]}
    model=build_rein();checked=[]
    with torch.inference_mode():
        for row in rows:
            scores=rein_outputs(model,tensor(row["crop_path"]))[0]
            x,y=row["crop_xy"]
            predicted=int(scores[0,:,y,x].argmax())
            checked.append(dict(point_id=row["point_id"],reference=expected[row["point_id"]],local=predicted))
    matches=sum(r["local"]==r["reference"] for r in checked)
    result=dict(status="exact_class_agreement" if matches==len(checked) else "differences_require_review",
        reference_checkpoint_sha256=fetched["sha256"],reference_gpu=old["gpu"],local_gpu=torch.cuda.get_device_name(),
        reference="Prior native-Meta-backbone + compiled-MMCV Modal inference on36 frozen NEOM points",
        local="Cached HF backbone converted to Meta parameter layout + MMCV PyTorch reference deformable attention",
        matching_predictions=matches,total=len(checked),points=checked)
    write_json(OUT/"rein_backend_verification.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="points"}),flush=True)
    if matches<len(checked)-1:raise RuntimeError("Local model differs materially from the recorded original inference")


if __name__=="__main__":main()
