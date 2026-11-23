"""Paper-based Rein-A reconstruction, including all six losses and SAM2 STM.

The authors have not released training code. In particular, SAM-mask query
encoding/alignment is underspecified. Our explicit implementation choices are
exported with every run; this module does not claim exact reproduction.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from .assets import OUT, write_json
from .probe import tensor
from .runtime import build_rein
from . import sam_masks


def semantic(classes,masks,size):
    scores=torch.einsum("bqc,bqhw->bchw",classes.softmax(-1)[...,:-1],masks.sigmoid())
    return F.interpolate(scores,size,mode="bilinear",align_corners=False)


def probability(scores):
    return scores.clamp_min(1e-8)/scores.sum(1,keepdim=True).clamp_min(1e-8)


def map_loss(scores,labels,weights=None):
    loss=F.nll_loss(probability(scores).log(),labels,ignore_index=0,reduction="none")
    valid=labels!=0
    if weights is not None:loss=loss*weights
    return loss.sum()/valid.sum().clamp_min(1)


def source_loss(head,classes,masks,labels):
    from mmseg.structures import SegDataSample
    from mmengine.structures import PixelData
    height,width=labels.shape[-2:]
    sample=SegDataSample(metainfo=dict(img_shape=(height,width),ori_shape=(height,width),
        pad_shape=(height,width),scale_factor=(1.,1.)))
    sample.gt_sem_seg=PixelData(data=labels[0,None])
    instances,metas=head._seg_data_to_instance_data([sample])
    losses=head.loss_by_feat(classes,masks,instances,metas)
    return sum(v for k,v in losses.items() if "loss" in k)


def instance_loss(student_classes,student_masks,teacher_classes,teacher_masks):
    teacher_classes=teacher_classes.detach();teacher_masks=teacher_masks.detach()
    labels=teacher_classes.softmax(-1).argmax(-1)
    class_weights=student_classes[-1].new_ones(97);class_weights[-1]=.1
    active=(labels!=96)&(labels!=0)
    total=student_classes[-1].sum()*0
    for classes,masks in zip(student_classes,student_masks):
        total=total+2*F.cross_entropy(classes.transpose(1,2),labels,weight=class_weights,ignore_index=0)
        if active.any():
            targets=F.interpolate(teacher_masks.sigmoid(),masks.shape[-2:],mode="bilinear",align_corners=False)[active]
            logits=masks[active]
            total=total+5*F.binary_cross_entropy_with_logits(logits,targets)
            values=logits.sigmoid().flatten(1);targets=targets.flatten(1)
            total=total+5*(1-(2*(values*targets).sum(1)+1)/(values.sum(1)+targets.sum(1)+1)).mean()
    return total


class SemanticTransfer(nn.Module):
    """Mask2Former STM: teacher features and SAM-mask-derived ordered queries.

    SAM regions are aligned by IoU to teacher queries so Eq19's direct query
    supervision has a consistent order. Mask-pooled features and a learned
    16x16 mask embedding supply the queries. These encoding/alignment choices
    are our reconstruction; the public paper does not specify their code.
    """
    def __init__(self,model):
        super().__init__()
        self.head=deepcopy(model.decode_head)
        self.feature_query=deepcopy(model.backbone.reins.transform)
        # Zero-init the new SAM query projections: the STM begins with the
        # source model's predictions, rather than supplying random supervision.
        nn.init.zeros_(self.feature_query.weight);nn.init.zeros_(self.feature_query.bias)
        self.mask_query=nn.Linear(256,256)
        nn.init.zeros_(self.mask_query.weight);nn.init.zeros_(self.mask_query.bias)
        self.requires_grad_(True)

    def forward(self,features,sam,teacher_masks):
        target=teacher_masks.sigmoid().detach()
        h,w=target.shape[-2:]
        if not len(sam):sam=target.new_ones((1,224,224))
        regions=F.interpolate(sam[:,None],(h,w),mode="nearest")[:,0]
        a=target[0].flatten(1)>.5;b=regions.flatten(1)>.5
        intersection=a.float()@b.float().T
        union=a.sum(1)[:,None]+b.sum(1)[None,:]-intersection
        order=(intersection/union.clamp_min(1)).argmax(1)
        ordered=regions[order]
        # Pool at the final teacher feature resolution.
        last=features["feature"][-1].detach()
        small=F.interpolate(ordered[:,None],last.shape[-2:],mode="nearest")[:,0]
        pooled=torch.einsum("qhw,bchw->bqc",small,last)/small.sum((-2,-1)).clamp_min(1)[None,:,None]
        geometry=F.interpolate(ordered[:,None],(16,16),mode="bilinear",align_corners=False).flatten(1)
        # Preserve distinct query identities when multiple teacher queries align
        # to the same SAM region (including single-region/empty-mask images).
        query=features["query"].detach()+self.feature_query(pooled)+self.mask_query(geometry)[None]
        inputs=dict(feature=[v.detach() for v in features["feature"]],query=query)
        return self.head(inputs)


class Teacher:
    def __init__(self,model,decay=.999):
        self.decay=decay
        self.shadow={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
                     if k.startswith(("backbone.reins.","decode_head."))}

    @contextmanager
    def apply(self,model):
        state=model.state_dict()
        saved={k:state[k].clone() for k in self.shadow}
        try:
            with torch.no_grad():
                for k,v in self.shadow.items():state[k].copy_(v)
            yield
        finally:
            with torch.no_grad():
                for k,v in saved.items():state[k].copy_(v)

    @torch.no_grad()
    def update(self,model,step):
        decay=min(self.decay,1-1/(step+1))
        for k,v in model.state_dict().items():
            if k not in self.shadow:continue
            value=v.detach().cpu()
            if value.is_floating_point():self.shadow[k].lerp_(value,1-decay)
            else:self.shadow[k].copy_(value)


def checkpoint_backbone(model):
    from torch.utils.checkpoint import checkpoint
    for block in model.backbone.blocks:
        original=block.forward
        def forward(*args,_original=original,**kwargs):
            if torch.is_grad_enabled():return checkpoint(_original,*args,use_reentrant=False,**kwargs)
            return _original(*args,**kwargs)
        block.forward=forward


def sample(row,size,flip=False,source=False):
    x=tensor(row["path"])
    masks=sam_masks.load(row["path"])
    if size!=224:
        x=F.interpolate(x,(size,size),mode="bilinear",align_corners=False)
        masks=F.interpolate(masks[:,None],(size,size),mode="nearest")[:,0] if len(masks) else masks.new_empty((0,size,size))
    labels=None
    if source:
        labels=torch.from_numpy(np.array(Image.open(row["mask"]),dtype=np.int64))[None].cuda()
        if size!=224:labels=F.interpolate(labels[:,None].float(),(size,size),mode="nearest")[:,0].long()
    if flip:
        x=x.flip(-1);masks=masks.flip(-1)
        if labels is not None:labels=labels.flip(-1)
    return x,masks,labels


def class_mix(xs,ys,xt,pseudo,source_masks,target_masks,quality):
    classes=ys.unique();classes=classes[classes!=0]
    classes=classes[torch.randperm(len(classes),device=classes.device)[:max(1,math.ceil(len(classes)/2))]]
    mask=(ys[:,None]==classes[None,:,None,None]).any(1)
    mixed=torch.where(mask[:,None],xs,xt)
    labels=torch.where(mask,ys,pseudo)
    weights=torch.where(mask,torch.ones_like(ys,dtype=torch.float32),torch.full_like(ys,quality,dtype=torch.float32))
    sam=torch.cat([source_masks*mask[0],target_masks*(~mask[0])])
    sam=sam[sam.sum((-2,-1))>0]
    return mixed,labels,weights,sam


def build():
    model=build_rein(trainable=True)
    checkpoint_backbone(model)
    stm=SemanticTransfer(model).cuda()
    teacher=Teacher(model)
    # Match the authors' norm/token/query weight-decay exceptions.
    groups=[]
    for prefix,network in [("student",model),("stm",stm)]:
        for module_name,module in network.named_modules():
            norm=isinstance(module,(nn.LayerNorm,nn.GroupNorm,nn.modules.batchnorm._BatchNorm))
            for name,p in module.named_parameters(recurse=False):
                if not p.requires_grad:continue
                full=f"{prefix}.{module_name}.{name}"
                no_decay=norm or any(k in full for k in ["norm","query_embed","level_embed",
                            "learnable_tokens","effective_ema_tokens","reins.scale"])
                groups.append(dict(params=[p],weight_decay=0. if no_decay else .05))
    optimizer=torch.optim.AdamW(groups,lr=1e-4,betas=(.9,.999),eps=1e-8)
    return model,stm,teacher,optimizer


def rare_class_sampler(source):
    counts=np.zeros(96,dtype=np.int64);members={i:[] for i in range(1,96)}
    for row in source:
        pixels=np.bincount(np.asarray(Image.open(row["mask"])).reshape(-1),minlength=96)
        counts+=pixels
        for i in range(1,96):
            if pixels[i]>=128:members[i].append(row)
    classes=[i for i in range(1,96) if members[i]]
    frequency=counts[classes]/max(counts[1:].sum(),1)
    logits=(1-frequency)/.01;weights=np.exp(logits-logits.max());weights/=weights.sum()
    def draw():
        category=random.choices(classes,weights=weights,k=1)[0]
        return random.choice(members[category])
    return draw


def strong(image):
    from torchvision.transforms import functional as TF
    mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
    std=image.new_tensor([.229,.224,.225])[None,:,None,None]
    value=(image*std+mean).clamp(0,1)
    if random.random()>.2:
        value=TF.adjust_brightness(value,random.uniform(.8,1.2))
        value=TF.adjust_contrast(value,random.uniform(.8,1.2))
        value=TF.adjust_saturation(value,random.uniform(.8,1.2))
        value=TF.adjust_hue(value,random.uniform(-.05,.05))
    if random.random()<.5:value=TF.gaussian_blur(value,[3,3],random.uniform(.15,1.15))
    return (value-mean)/std


def update(model,stm,teacher,optimizer,source,target,step,accumulation=1,size=224,sampler=None):
    optimizer.zero_grad(set_to_none=True)
    records=Counter()
    for micro in range(accumulation):
        sr=sampler() if sampler is not None else source[(step*accumulation+micro)%len(source)]
        tr=target[(step*accumulation+micro)%len(target)]
        xs,ms,ys=sample(sr,size,random.random()<.5,source=True)
        xt,mt,_=sample(tr,size,random.random()<.5)
        model.eval();stm.eval()
        with teacher.apply(model),torch.no_grad():
            ft=model.backbone(xt);ct,bt=model.decode_head(ft)
            probabilities=probability(semantic(ct[-1],bt[-1],xt.shape[-2:]))
            confidence,pseudo=probabilities.max(1)
            quality=float((confidence>=.968).float().mean())
            xm,ym,weights,mm=class_mix(xs,ys,xt,pseudo,ms,mt,quality)
            fs=model.backbone(xs);cs,bs=model.decode_head(fs)
            xm=strong(xm)
            fm=model.backbone(xm);cm,bm=model.decode_head(fm)
            cstm,mstm=stm(ft,mt,bt[-1])
            if step==0 and micro==0:
                delta=max(float((cstm[-1]-ct[-1]).abs().max()),float((mstm[-1]-bt[-1]).abs().max()))
                if delta>1e-4:raise RuntimeError(f"STM warm-start changed initial teacher predictions: {delta}")
                records["stm_initial_max_abs"]=delta
        model.train();stm.train()
        # Each branch is backpropagated separately to bound peak memory.
        sf=model.backbone(xs);sc,sm=model.decode_head(sf)
        loss=source_loss(model.decode_head,sc,sm,ys)
        (loss/accumulation).backward();records["source"]+=float(loss.detach())/accumulation
        del sf,sc,sm,loss
        sc,sm=stm(fs,ms,bs[-1]);loss=source_loss(stm.head,sc,sm,ys)
        (loss/accumulation).backward();records["stm_source"]+=float(loss.detach())/accumulation
        del sc,sm,loss,fs,cs,bs
        mf=model.backbone(xm);mc,mb=model.decode_head(mf)
        loss=map_loss(semantic(mc[-1],mb[-1],xm.shape[-2:]),ym,weights)
        (loss/accumulation).backward();records["mix"]+=float(loss.detach())/accumulation
        del mf,mc,mb,loss
        mc,mb=stm(fm,mm,bm[-1]);loss=map_loss(semantic(mc[-1],mb[-1],xm.shape[-2:]),ym,weights)
        (loss/accumulation).backward();records["stm_mix"]+=float(loss.detach())/accumulation
        del mc,mb,loss,fm,cm,bm
        blocks=torch.rand((1,1,math.ceil(size/64),math.ceil(size/64)),device=xt.device)>.7
        keep=F.interpolate(blocks.float(),(size,size),mode="nearest")
        masked=model.backbone(xt*keep);cc,bb=model.decode_head(masked)
        loss_a=instance_loss(cc,bb,ct[-1],bt[-1])
        loss_b=instance_loss(cc,bb,cstm[-1],mstm[-1])
        ((loss_a+loss_b)/accumulation).backward()
        records["mask"]+=float(loss_a.detach())/accumulation
        records["stm_mask"]+=float(loss_b.detach())/accumulation
        records["pseudo_confident_fraction"]+=quality/accumulation
        del masked,cc,bb,loss_a,loss_b,ft,ct,bt,cstm,mstm
    parameters=[p for group in optimizer.param_groups for p in group["params"]]
    if any(p.grad is not None for name,p in model.backbone.named_parameters() if not name.startswith("reins.")):
        raise RuntimeError("A frozen foundation-backbone weight received a gradient")
    norm=torch.nn.utils.clip_grad_norm_(parameters,max_norm=.01,error_if_nonfinite=True)
    optimizer.step();teacher.update(model,step)
    records["gradient_norm"]=float(norm)
    if not all(math.isfinite(v) for v in records.values()):raise RuntimeError("Nonfinite adaptation loss")
    return dict(records)


def main():
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--size",type=int,default=512)
    parser.add_argument("--accumulation",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--probe",action="store_true")
    args=parser.parse_args()
    if not 1<=args.steps<=40000 or args.size not in [224,512] or args.accumulation<1:
        raise ValueError("Use 1..40000 updates, 224/512 input, and positive accumulation")
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    source_path=OUT/("source512.json" if args.size==512 else "source.json")
    source=json.loads(source_path.read_text())
    target=json.loads((OUT/"adaptation.json").read_text())
    if any("label" in r or "group" in r for r in target):raise RuntimeError("Target labels leaked into adaptation inputs")
    random.Random(42).shuffle(source);random.Random(42).shuffle(target)
    directory=OUT/"adaptation";directory.mkdir(exist_ok=True)
    if not args.probe and not args.resume and (directory/"final.pt").exists():
        raise RuntimeError("Completed adaptation checkpoint exists; refusing an accidental retrain")
    model,stm,teacher,optimizer=build()
    sampler=rare_class_sampler(source)
    start=0
    if args.resume:
        checkpoint=torch.load(directory/"last.pt",map_location="cpu",weights_only=True)
        model.load_state_dict(checkpoint["model"],strict=False)
        stm.load_state_dict(checkpoint["stm"],strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        teacher.shadow=checkpoint["teacher"]
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start=checkpoint["step"]
    if not args.probe:
        if (directory/"completed.json").exists():(directory/"completed.json").unlink()
        write_json(directory/"config.json",dict(arguments=vars(args),source_images=len(source),target_images=len(target),
            paper="https://arxiv.org/abs/2508.01667",algorithm="Algorithm1, equations14-20, all six losses",
            source_checkpoint="User's exact Rein++ MH4/GELU + Mask2Former, selected epoch40 EMA",
            teacher_decay=.999,lr=1e-4,weight_decay=.05,poly_horizon=40000,poly_power=.9,
            alpha=1.,beta=1.,pseudo_threshold=.968,mask_ratio=.7,mask_block=64,
            sam="facebook/sam2-hiera-tiny",rcs_temperature=.01,effective_batch=args.accumulation,
            target_labels_used_for_training=False,checkpoint_selection="Fixed update count; student branch; no target test selection",
            fidelity="Paper-based reconstruction; not an exact reproduction of unreleased author training code",
            documented_choices=["Small preselected source/target subsets for a first local comparison",
                "Native224 RSG patches resized to512; source crops native512",
                "SAM2 Hiera-T; paper does not specify its SAM2 backbone size",
                "STM query implementation: SAM-mask pooling plus zero-initialized feature/mask projections, teacher query prior and IoU alignment; absent from released code",
                "If SAM finds no region, use a neutral full-image mask query; empty-mask coverage is reported",
                "EMA checkpoint's effective token matrices kept as dense trainable parameters to preserve exact starting predictions",
                "Physical batch1 with gradient accumulation; differentiable PyTorch reference deformable attention",
                "Source branch uses original published Mask2Former Hungarian loss; masked branches use query-aligned CE/BCE/Dice with auxiliary-head supervision",
                "Mixed SAM masks constructed from offline source/target masks to avoid SAM inference inside training"] ))
    for step in range(start,3 if args.probe else args.steps):
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        # Match the published polynomial schedule horizon; shortened pilots do
        # not silently compress a 40000-step learning-rate schedule.
        rate=1e-4*(1-step/40000)**.9
        for group in optimizer.param_groups:group["lr"]=rate
        losses=update(model,stm,teacher,optimizer,source,target,step,args.accumulation,args.size,sampler)
        torch.cuda.synchronize()
        record=dict(step=step+1,seconds=time.monotonic()-started,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                    lr=rate,losses=losses,probe=args.probe)
        print(json.dumps(record),flush=True)
        if args.probe:write_json(OUT/"adaptation_probe.json",record)
        else:
            with (directory/"history.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
            write_json(directory/"progress.json",record)
            if (step+1)%25==0 or step+1==args.steps:
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
                       if k.startswith(("backbone.reins.","decode_head."))}
                temporary=directory/"final.tmp"
                torch.save(dict(model=state,step=step+1,configuration=vars(args),kind="Rein-A paper-based reconstruction, student inference"),temporary)
                temporary.replace(directory/("final.pt" if step+1==args.steps else "current.pt"))
                temporary=directory/"last.tmp"
                torch.save(dict(model=state,stm=stm.state_dict(),optimizer=optimizer.state_dict(),teacher=teacher.shadow,
                    step=step+1,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),configuration=vars(args)),temporary)
                temporary.replace(directory/"last.pt")
    if args.probe:print("All six Rein-A losses backpropagated successfully",flush=True)
    else:write_json(directory/"completed.json",dict(status="completed",updates=args.steps,
        effective_batch=args.accumulation,input_size=args.size,all_six_losses=True,
        frozen_backbone_gradients=False,student_checkpoint=str(directory/"final.pt")))


if __name__=="__main__":main()
