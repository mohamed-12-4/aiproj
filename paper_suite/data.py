"""Lossless native V2 extraction into the authors' paired-folder interface."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from .recipe import DATA_REVISION, data_directory, root, write_json


def prepare():
    source = root() / "data" / ("coralscapes_v2_" + DATA_REVISION[:12]) / "data"
    destination = data_directory()
    ready = destination / "ready.json"
    expected = {"train": 1790, "val": 170, "test": 473}
    if ready.exists():
        metadata = json.loads(ready.read_text())
        if metadata.get("revision") == DATA_REVISION and metadata.get("counts") == expected:
            print("Reusing verified native V2 files", flush=True)
            return metadata
    cached_counts = {split: sum(pq.read_metadata(p).num_rows for p in source.glob(split + "-*.parquet"))
                     for split in ("train", "validation", "test")}
    if cached_counts != {"train": 1790, "validation": 170, "test": 473}:
        from huggingface_hub import snapshot_download
        snapshot_download("josauder/coralscapesV2", repo_type="dataset", revision=DATA_REVISION,
                          local_dir=source.parent, allow_patterns=["data/*.parquet", "id2label.json"],
                          max_workers=4)
    manifest, counts = [], Counter()
    for source_split, split in (("train", "train"), ("validation", "val"), ("test", "test")):
        shards = sorted(source.glob(source_split + "-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No {source_split} shards in {source}")
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=1, columns=["image", "label"], use_threads=False):
                row = batch.to_pylist()[0]
                filename = Path(row["image"]["path"]).stem
                # Glob pairing is suffix-based; retain original encoded bytes,
                # independent of extension, and let Pillow inspect the header.
                site = re.search(r"site\d+", filename)
                site = site[0] if site else "unknown"
                img_path = destination / "leftImg8bit" / split / site / (filename + "_leftImg8bit.png")
                mask_path = destination / "gtFine" / split / site / (filename + "_gtFine.png")
                with Image.open(io.BytesIO(row["image"]["bytes"])) as image, Image.open(io.BytesIO(row["label"]["bytes"])) as mask:
                    if image.size != mask.size or image.size != (2048, 1024):
                        raise ValueError("Native V2 image/mask geometry mismatch")
                    ids = np.asarray(mask)
                    if ids.min() < 0 or ids.max() > 95 or ids.ndim != 2:
                        raise ValueError("Unexpected V2 native semantic IDs")
                    image_format = image.format
                for path, key in ((img_path, "image"), (mask_path, "label")):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = row[key]["bytes"]
                    if not path.exists() or path.stat().st_size != len(payload):
                        temp = path.with_suffix(".tmp")
                        temp.write_bytes(payload)
                        temp.replace(path)
                    elif hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
                        raise ValueError(f"Cached file content differs: {path}")
                manifest.append({"split": split, "site": site, "source_path": row["image"]["path"],
                                 "image": str(img_path.relative_to(destination)), "label": str(mask_path.relative_to(destination)),
                                 "image_sha256": hashlib.sha256(row["image"]["bytes"]).hexdigest(),
                                 "label_sha256": hashlib.sha256(row["label"]["bytes"]).hexdigest(),
                                 "image_encoding": image_format})
                counts[split] += 1
                if sum(counts.values()) % 200 == 0:
                    print(f"Prepared {sum(counts.values())}/2433 native V2 pairs", flush=True)
    if dict(counts) != expected:
        raise ValueError(f"Incomplete V2 data: {dict(counts)}")
    # Raw source IDs remain unchanged; void=0, named Background has its own ID.
    write_json(destination / "manifest.json", manifest)
    metadata = {"revision": DATA_REVISION, "counts": dict(counts), "native_size_wh": [2048, 1024],
                "num_logits": 96, "ignore_index": 0, "native_label_ids_preserved": True,
                "no_image_resampling_or_reencoding": True}
    write_json(ready, metadata)
    return metadata
