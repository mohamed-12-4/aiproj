"""Evaluate native predictions before mapping; select LiNeS on validation only."""
import argparse
from collections import defaultdict
import csv
import gc
import json
from pathlib import Path
import time
import hashlib

import numpy as np
import torch

from .assets import OUT, write_json, sha256
from .labels import GROUPS, native_lookup
from .probe import tensor
from .runtime import build_dpt, build_rein, rein_outputs


def metrics(rows, key="predicted_group"):
    cm=np.zeros((len(GROUPS),len(GROUPS)),dtype=np.int64)
    for r in rows:cm[r["group"],r[key]]+=1
    by_class={}
    for i,name in enumerate(GROUPS):
        support=int(cm[i].sum())
        if not support:continue
        predicted=int(cm[:,i].sum());tp=int(cm[i,i])
        by_class[name]=dict(support=support,recall=tp/support,
            precision=tp/predicted if predicted else None,f1=2*tp/(support+predicted))
    return dict(samples=len(rows),accuracy=float(cm.trace()/cm.sum()),
        macro_recall=float(np.mean([v["recall"] for v in by_class.values()])),
        macro_f1=float(np.mean([v["f1"] for v in by_class.values()])),
        classes=by_class,confusion=cm.tolist(),order=GROUPS)


def run(model, label, split, method, gamma=None,size=512):
    path=OUT/f"{split}.json"
    rows=json.loads(path.read_text())
    suffix="" if size==224 else f"_s{size}"
    result_path=OUT/"results"/f"{label}{suffix}_{split}.json"
    lookup=native_lookup()
    mapping_hash=hashlib.sha256(json.dumps(lookup).encode()).hexdigest()
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old["split_sha256"]!=sha256(path):raise RuntimeError("Evaluation inputs changed")
        if old.get("mapping_sha256")!=mapping_hash:
            old.setdefault("mapping_history",[]).append(dict(metrics=old["metrics"],
                reason="Ontology review: include Turbinaria in hard coral and transect lines in human objects"))
            for r in old["predictions"]:
                r["predicted_group"]=lookup[r["predicted_native"]]
                r["patch_average_group"]=lookup[r["patch_average_native"]]
            old["metrics"]=metrics(old["predictions"])
            old["patch_average_sensitivity"]=metrics(old["predictions"],"patch_average_group")
            groups=defaultdict(list)
            for r in old["predictions"]:groups[r["site"]].append(r)
            old["by_site"]={k:metrics(v) for k,v in groups.items()}
            old["mapping_sha256"]=mapping_hash
            write_json(result_path,old)
        print("Reusing completed evaluation",label,split,flush=True);return old
    started=time.monotonic();predictions=[]
    model.eval();torch.cuda.reset_peak_memory_stats()
    examples={next(r["name"] for r in rows if r["group"]==group) for group in sorted({r["group"] for r in rows})}
    with torch.inference_mode():
        for i,row in enumerate(rows):
            if sha256(row["path"])!=row["sha256"]:raise RuntimeError("An evaluation input changed")
            x=tensor(row["path"])
            if size!=224:x=torch.nn.functional.interpolate(x,(size,size),mode="bilinear",align_corners=False)
            if method=="released":
                with torch.autocast("cuda",dtype=torch.bfloat16):scores=model(x)
            else:scores=rein_outputs(model,x)[0]
            if size!=224:scores=torch.nn.functional.interpolate(scores,(224,224),mode="bilinear",align_corners=False)
            if not torch.isfinite(scores).all():raise RuntimeError("Nonfinite predictions")
            center=int(scores[0,:,112,112].argmax())
            probabilities=scores.float().softmax(1) if method=="released" else scores.float()/scores.float().sum(1,keepdim=True).clamp_min(1e-8)
            average=int(probabilities[0].mean((-2,-1)).argmax())
            predictions.append(dict(row,predicted_native=center,predicted_group=lookup[center],
                                    patch_average_native=average,patch_average_group=lookup[average]))
            if split=="test" and row["name"] in examples:
                directory=OUT/"prediction_maps"/f"s{size}"/label;directory.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(directory/(row["name"]+".npz"),native=scores[0].argmax(0).byte().cpu().numpy())
            if (i+1)%32==0:print(f"{label} {split} {i+1}/{len(rows)}",flush=True)
    groups=defaultdict(list)
    for r in predictions:groups[r["site"]].append(r)
    result=dict(model=label,method=method,gamma=gamma,split=split,split_sha256=sha256(path),mapping_sha256=mapping_hash,
        seconds=time.monotonic()-started,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        metrics=metrics(predictions),patch_average_sensitivity=metrics(predictions,"patch_average_group"),
        by_site={k:metrics(v) for k,v in groups.items()},predictions=predictions,
        input_size=size,
        protocol=f"224x224 native patch resized to {size} square, batch1; scores mapped back to native224 before native96 argmax at center-pixel patch-label proxy and group mapping; original point transform unavailable; no target test labels used for adaptation",
        batchnorm_policy="Published DPT buffers retained identically for direct and LiNeS models")
    write_json(result_path,result)
    print(json.dumps(dict(model=label,split=split,metrics={k:result["metrics"][k] for k in ["samples","accuracy","macro_recall","macro_f1"]},seconds=result["seconds"])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("action",choices=["validation","test","adapted-validation","adapted-test","adapted-all"])
    parser.add_argument("--size",type=int,default=512)
    parser.add_argument("--selection-size",type=int,default=512)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.action=="validation":
        candidates=[]
        for gamma in [1.,.25,.5,.75]:
            model=build_dpt(gamma)
            label="released" if gamma==1 else f"lines_{gamma:g}"
            result=run(model,label,"validation","released",gamma,args.size)
            candidates.append(dict(gamma=gamma,macro_recall=result["metrics"]["macro_recall"]))
            del model;gc.collect();torch.cuda.empty_cache()
        winner=max(candidates,key=lambda r:r["macro_recall"])
        best_edit=max([r for r in candidates if r["gamma"]<1],key=lambda r:r["macro_recall"])
        write_json(OUT/f"lines_selection_s{args.size}.json",dict(candidates=candidates,selected=winner,
            best_nontrivial_edit=best_edit,
            selection_split="validation",criterion="common-group macro recall; gamma1 is the no-edit control"))
        model=build_rein();run(model,"reinpp40","validation","rein",size=args.size)
    elif args.action=="test":
        selection=json.loads((OUT/f"lines_selection_s{args.selection_size}.json").read_text())
        selected=max([r for r in selection["candidates"] if r["gamma"]<1],key=lambda r:r["macro_recall"])["gamma"]
        for gamma,label in [(1.,"released"),(selected,"lines_selected")]:
            model=build_dpt(gamma);run(model,label,"test","released",gamma,args.size)
            del model;gc.collect();torch.cuda.empty_cache()
        model=build_rein();run(model,"reinpp40","test","rein",size=args.size)
    else:
        if not (OUT/"adaptation/completed.json").exists():raise RuntimeError("Adaptation has not completed its planned updates")
        model=build_rein()
        state=torch.load(OUT/"adaptation/final.pt",map_location="cpu",weights_only=True)
        loaded=model.load_state_dict(state["model"],strict=False)
        if loaded.unexpected_keys or any(not k.startswith("backbone.") for k in loaded.missing_keys):
            raise RuntimeError("Adapted inference export mismatch")
        if args.action=="adapted-all":
            for split,size in [("validation",512),("test",512),("test",224)]:
                run(model,"rein_a",split,"rein",size=size)
        else:
            split="validation" if args.action=="adapted-validation" else "test"
            run(model,"rein_a",split,"rein",size=args.size)


if __name__=="__main__":main()
