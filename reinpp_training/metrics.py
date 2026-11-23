"""Native V2 and fixed health mapping, independent of the Transformers runtime."""
import json
from pathlib import Path
import torch


def finite(value):
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite(v) for v in value]
    if isinstance(value,float) and not __import__('math').isfinite(value):return None
    return value


def summarize(counts,ignore=0):
    c=counts.double();tp=c.diag();union=c.sum(0)+c.sum(1)-tp
    iou=torch.where(union>0,tp/union.clamp_min(1),torch.nan)
    valid=union>0
    if ignore is not None:valid[ignore]=False
    return finite(dict(miou=float(iou[valid].mean()),pixel_accuracy=float(tp.sum()/c.sum().clamp_min(1)),
                       per_class_iou=iou.tolist(),confusion=counts.long().tolist()))


def mapped_metrics(counts):
    p=Path(__file__).resolve().parents[1]/"coralscapes_metadata/v2.json"
    meta=json.loads(p.read_text())
    legacy=torch.zeros(96,dtype=torch.long,device=counts.device)
    for source,target in meta["fine_to_legacy_id"].items():legacy[int(source)]=int(target)
    lookup=torch.zeros(40,dtype=torch.long,device=counts.device)
    lookup[0]=lookup[21]=lookup[27]=4
    for target,ids in ((1,[6,17,22,25,28,31,34,36]),(2,[4,16,19,33]),(3,[3,20,23,32,37])):lookup[ids]=target
    health=lookup[legacy];valid=health!=4
    positions=(health[valid,None]*5+health[None,:]).flatten()
    hc=torch.zeros(25,dtype=torch.long,device=counts.device)
    hc.scatter_add_(0,positions,counts[valid].flatten());hc=hc.reshape(5,5)
    result=summarize(hc,ignore=4)
    result["coral_miou"]=finite(float(torch.tensor([v if v is not None else float('nan') for v in result["per_class_iou"][1:4]]).nanmean()))
    tp=hc[3,3].double()
    result["dead"]=dict(precision=float(tp/hc[:,3].sum().clamp_min(1)),recall=float(tp/hc[3].sum().clamp_min(1)),iou=result["per_class_iou"][3])
    coarse=torch.zeros(1600,dtype=torch.long,device=counts.device)
    coarse.scatter_add_(0,(legacy[:,None]*40+legacy[None,:]).flatten(),counts.flatten());coarse=coarse.reshape(40,40)
    result["legacy_dead_recalls"]={str(i):float(coarse[i,lookup==3].sum()/coarse[i].sum()) if coarse[i].sum() else None for i in (3,20,23,32,37)}
    result["fine_dead_recalls"]={name:float(counts[int(i),health==3].sum()/counts[int(i)].sum()) if counts[int(i)].sum() else None
                                 for i,name in meta["id2label"].items() if int(health[int(i)])==3}
    return result
