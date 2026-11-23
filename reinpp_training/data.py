"""Authors' MMSeg augmentation policy over verified native V2 image/mask pairs."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from .config import DATA_REVISION, PROJECT, train_pipeline

DATA=Path("/workspace/data")/("coralscapes_v2_paper_"+DATA_REVISION[:12])


def register_dataset():
    from mmseg.registry import DATASETS
    from mmseg.datasets import BaseSegDataset
    if DATASETS.get("NativeCoralscapesV2") is not None:return
    meta=json.loads((PROJECT/"coralscapes_metadata/v2.json").read_text())
    classes=("void",)+tuple(meta["id2label"][str(i)] for i in range(1,96))
    @DATASETS.register_module()
    class NativeCoralscapesV2(BaseSegDataset):
        METAINFO=dict(classes=classes,palette=[[(i*53)%256,(i*97)%256,(i*193)%256] for i in range(96)])
        def __init__(self,**kwargs):
            super().__init__(img_suffix="_leftImg8bit.png",seg_map_suffix="_gtFine.png",
                             reduce_zero_label=False,ignore_index=0,**kwargs)


def build_dataset(split="train"):
    from mmseg.registry import DATASETS
    register_dataset()
    pipeline=train_pipeline() if split=="train" else [dict(type="LoadImageFromFile"),
                    dict(type="LoadAnnotations",reduce_zero_label=False,imdecode_backend="pillow"),dict(type="PackSegInputs")]
    dataset=DATASETS.build(dict(type="NativeCoralscapesV2",data_root=str(DATA),
                          data_prefix=dict(img_path=f"leftImg8bit/{split}",seg_map_path=f"gtFine/{split}"),
                          pipeline=pipeline,test_mode=split!="train"))
    expected={"train":1790,"val":170,"test":473}
    if len(dataset)!=expected[split]:raise ValueError(f"Unexpected {split} pair count: {len(dataset)}")
    return dataset


def worker_seed(worker):
    seed=torch.initial_seed()%(2**32)
    np.random.seed(seed);random.seed(seed)


def train_loader(dataset,epoch,workers=8):
    from mmengine.dataset import pseudo_collate
    generator=torch.Generator().manual_seed(42+epoch)
    return torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=True,num_workers=workers,
                collate_fn=pseudo_collate,pin_memory=True,drop_last=True,worker_init_fn=worker_seed,
                generator=generator,**({"prefetch_factor":2} if workers else {}))
