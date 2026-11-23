"""Single-image local GPU checks before any adaptation training."""
import argparse
import gc
import json
import time

import numpy as np
from PIL import Image
import torch

from .assets import OUT, write_json
from .runtime import build_dpt, build_rein, rein_outputs


def tensor(path, device="cuda"):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    image = torch.from_numpy(image.copy()).permute(2,0,1)
    image = (image - torch.tensor([.485,.456,.406])[:,None,None]) / torch.tensor([.229,.224,.225])[:,None,None]
    return image.unsqueeze(0).to(device)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--model",choices=["released","rein"],required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("The local CUDA GPU is not available")
    path=next((OUT/"patches").glob("*.png"))
    started=time.monotonic()
    model=build_dpt() if args.model=="released" else build_rein()
    torch.cuda.reset_peak_memory_stats()
    x=tensor(path)
    with torch.inference_mode():
        if args.model=="released":
            with torch.autocast("cuda",dtype=torch.bfloat16): output=model(x)
        else: output=rein_outputs(model,x)[0]
    torch.cuda.synchronize()
    if output.shape!=(1,96,224,224) or not torch.isfinite(output).all():
        raise RuntimeError(f"Invalid segmentation output {output.shape}")
    result=dict(status="passed",model=args.model,gpu=torch.cuda.get_device_name(),
                seconds=time.monotonic()-started,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                shape=list(output.shape),center_native=int(output[0,:,112,112].argmax()))
    write_json(OUT/(args.model+"_probe.json"),result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
