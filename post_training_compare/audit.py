"""Verify input isolation and checksums before model adaptation and test scoring."""
import json
import re
from collections import Counter

from .assets import OUT, write_json, sha256
from .labels import native_lookup


def main():
    splits={k:json.loads((OUT/f"{k}.json").read_text()) for k in ["adaptation","validation","test"]}
    sites={k:{r["site"] for r in rows} for k,rows in splits.items()}
    photos={k:{re.sub(r"_\d+\.png$","",r["name"]) for r in rows} for k,rows in splits.items()}
    for a,b in [("adaptation","validation"),("adaptation","test"),("validation","test")]:
        if sites[a]&sites[b] or photos[a]&photos[b]:raise RuntimeError("Target split leakage")
    for key,rows in splits.items():
        if len(photos[key])!=len(rows):raise RuntimeError("Multiple patches from the same photo")
        for row in rows:
            if sha256(row["path"])!=row["sha256"]:raise RuntimeError("Target input checksum changed")
            if key=="adaptation" and ("label" in row or "group" in row):raise RuntimeError("Target training labels leaked")
    source=json.loads((OUT/"source512.json").read_text())
    for row in source:
        if sha256(row["path"])!=row["sha256"] or sha256(row["mask"])!=row["mask_sha256"]:
            raise RuntimeError("Source input checksum changed")
    lookup=native_lookup()
    result=dict(status="passed",samples={k:len(v) for k,v in splits.items()},
        site_prefix_groups={k:len(v) for k,v in sites.items()},source_images=len(source),
        target_labels_available_to_training=False,original_photos_disjoint=True,
        split_hashes={k:sha256(OUT/f"{k}.json") for k in splits},native_to_group=lookup,
        caveats=["Geographic identity inferred from filename prefixes; no independent GPS audit",
                 "No exhaustive duplicate screening against all Coralscapes training images",
                 "Center-pixel readout is a proxy for the supplied patch label; crop-to-original-point transform is not supplied",
                 "Small stratified patch-label benchmark, not dense segmentation evaluation or original ReefNet protocol"])
    write_json(OUT/"input_audit.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
