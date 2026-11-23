"""Short measured cost/step screen; does not modify any training checkpoint."""

import argparse
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .adapters import build_model
from .ema import EffectiveEMA
from .execution import precision_dtype
from .recipe import make_config, root, write_json
from .runner import author, base_dataset, model_kwargs, state_fingerprint
from dinov3_dpt_seg.augmentations import SegmentationTrainAugmentation, segmentation_augment_config_from_dict


def benchmark(group, precision, steps=6, method="lora", decoder_checkpointing=True, commit=lambda: None):
    cfg = make_config(method, group)
    cfg["experiment"].update(precision=precision, decoder_checkpointing=decoder_checkpointing)
    author._set_seed(42)
    device = torch.device("cuda")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        return {"status": "unsupported", "precision": precision}
    model = build_model(model_kwargs(cfg), cfg["experiment"], device=device)
    decoder_hash = state_fingerprint(model, "neck.") + state_fingerprint(model, "seg_head.")
    if method == "eva":
        from .eva import initialize_eva
        initialize_eva(model, cfg, Path(cfg["train"]["output_dir"]), commit=commit)
    observed_dtypes = {}
    linear = next(m for m in model.modules() if isinstance(m,torch.nn.Linear))
    linear.register_forward_hook(lambda m,a,out: observed_dtypes.update(linear_output=str(out.dtype)))
    ema = EffectiveEMA(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=.01)
    optimizer.register_step_post_hook(lambda *_: ema.update(model))
    scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16")
    aug = SegmentationTrainAugmentation(segmentation_augment_config_from_dict(cfg["augment"]),
                                       model.image_mean, model.image_std, ignore_index=0)
    dataset = base_dataset(cfg,"train")
    size = tuple(max(cfg["augment"]["train_sizes"], key=lambda hw: hw[0]*hw[1]))
    batches=[aug(*dataset[i],target_size=size) for i in range(8)]
    x=torch.stack([v[0] for v in batches]).cuda()
    y=torch.stack([v[1] for v in batches]).cuda()
    del batches
    times=[]
    model.train()
    readout_errors = []
    if method == "rein_g":
        captured, handles = {}, []
        for index in model.backbone_layer_indices:
            def capture(module, args, output, i=index):
                captured[i] = output.detach()
            handles.append(model.encoder.layer[index].register_forward_hook(capture))
        with torch.no_grad(), torch.autocast("cuda", dtype=precision_dtype(cfg)):
            states = model.encoder(x[:1], output_hidden_states=True).hidden_states
            for index, feature in zip(model.backbone_layer_indices, model._select_hidden_states(states)):
                expected = torch.cat((captured[index][:,:1], captured[index][:,5:]),dim=1)
                error = float((feature-expected).abs().max())
                readout_errors.append(error)
                if error != 0:
                    raise RuntimeError(f"DPT missed the corrected Rein output at block {index}")
        for handle in handles:
            handle.remove()
        del captured, states, feature, expected
    torch.cuda.reset_peak_memory_stats()
    rein_gradient_norms = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started=time.perf_counter()
        with torch.autocast("cuda",dtype=precision_dtype(cfg)):
            logits=model(x)
            loss=F.cross_entropy(logits,y,ignore_index=0)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite benchmark loss")
        actual_dtype=str(logits.dtype)
        scaler.scale(loss).backward()
        if method == "rein_g":
            norms = model.encoder.reins.learnable_tokens.grad.flatten(1).norm(dim=1)
            if not torch.isfinite(norms).all() or not (norms > 0).all():
                raise RuntimeError("Every corrected Rein block must receive finite nonzero gradients")
            rein_gradient_norms = norms.detach().cpu().tolist()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        if step>=2:
            times.append(elapsed)
        print(f"BENCH {torch.cuda.get_device_name()} {precision} step={step} seconds={elapsed:.3f}",flush=True)
        del logits,loss
    model.eval()
    with torch.no_grad(),ema.apply(model),torch.autocast("cuda",dtype=precision_dtype(cfg)):
        outputs=model(x[:1])
        if not torch.isfinite(outputs).all():
            raise RuntimeError("Invalid EMA benchmark inference")
    return {"status":"passed","method":method,"gpu":torch.cuda.get_device_name(),"precision":precision,
            "actual_logits_dtype":actual_dtype,"physical_batch":8,"shape_hw":size,
            "observed_compute_dtype":observed_dtypes,
            "decoder_checkpointing":decoder_checkpointing,"bn_recompute_buffer_policy":"cloned buffers, one persistent update per batch",
            "decoder_initialization_sha256":decoder_hash,"rein_readout_max_abs":readout_errors,
            "rein_token_gradient_norm_by_layer":rein_gradient_norms,
            "seconds_per_step_median":statistics.median(times),"timed_steps":times,
            "optimizer_updates":ema.num_updates,"peak_allocated_GiB":torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_GiB":torch.cuda.max_memory_reserved()/2**30}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--group",required=True)
    parser.add_argument("--precision",choices=["bf16","fp16"],required=True)
    parser.add_argument("--method",choices=["lora","rein_g"],default="lora")
    parser.add_argument("--decoder-checkpointing",choices=["0","1"],default="1")
    args=parser.parse_args()
    try:
        result=benchmark(args.group,args.precision,method=args.method,decoder_checkpointing=args.decoder_checkpointing=="1")
    except Exception as error:
        result={"status":"failed","precision":args.precision,"error_type":type(error).__name__,"error":str(error)[:1800]}
        print(result,flush=True)
    write_json(root()/"checkpoints"/args.group/(args.precision+".json"),result)
