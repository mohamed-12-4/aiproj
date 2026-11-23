"""Inspect and selectively read the RSG ZIP by HTTP ranges; never fetch it whole."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile
from collections import OrderedDict
from collections import Counter, defaultdict
import hashlib
import random

from .assets import OUT, DATASET, write_json
from .assets import sha256
from .labels import target_group, GROUPS, native_lookup


def annotations():
    assets = json.loads((OUT / "assets.json").read_text())
    path = assets["dataset"]["local"]["ReefNet_RSG-TestDataset_annotations.csv"]
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class HTTPRangeFile(io.RawIOBase):
    """Bounded range reader with known size, avoiding HF directory API calls."""
    def __init__(self, url, size):
        import requests
        self.url, self.size, self.position = url, size, 0
        self.session = requests.Session()
        self.blocks = OrderedDict()
        self.block_size, self.downloaded = 1 << 18, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0: raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0: size = self.size - self.position
        size = min(size, self.size - self.position)
        result = bytearray()
        while size > 0:
            number, offset = divmod(self.position, self.block_size)
            if number not in self.blocks:
                start = number * self.block_size
                end = min(start + self.block_size, self.size) - 1
                # Range in the query also prevents intermediaries from caching
                # one range response and incorrectly serving it for another.
                response = self.session.get(self.url + f"?download=true&range_start={start}",
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                    timeout=60,stream=True)
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{end}":
                    raise RuntimeError("Remote host did not honor the requested byte range; refusing a whole-ZIP download")
                data=response.raw.read(end-start+2)
                response.close()
                if len(data)!=end-start+1:raise RuntimeError("Incorrect remote range length")
                self.blocks[number] = data
                self.downloaded += len(data)
                if len(self.blocks) > 32: self.blocks.popitem(last=False)
            block = self.blocks[number]
            n = min(size, len(block) - offset)
            if n <= 0: raise EOFError("Incomplete remote ZIP block")
            result.extend(block[offset:offset+n])
            self.position += n
            size -= n
        return bytes(result)

    def close(self):
        self.session.close()
        super().close()


def remote_zip():
    assets = json.loads((OUT / "assets.json").read_text())
    revision = assets["dataset"]["revision"]
    name = "ReefNet_RSG-TestDataset_images.zip"
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}"
    file = HTTPRangeFile(url, assets["dataset"]["files"][name]["bytes"])
    return file, zipfile.ZipFile(file)


def inspect():
    from PIL import Image
    rows = annotations()
    file, archive = remote_zip()
    with file, archive:
        infos = [v for v in archive.infolist() if not v.is_dir()]
        print(json.dumps(dict(zip_members=len(infos), sample_members=[dict(name=v.filename,
            compressed=v.compress_size, uncompressed=v.file_size) for v in infos[:8]])), flush=True)
        lookup = {Path(v.filename).name: v.filename for v in infos}
        for row in rows[:3]:
            entry = lookup[row["name"]]
            data = archive.read(entry)
            image = Image.open(io.BytesIO(data))
            print(json.dumps(dict(row=row, member=entry, size=image.size, mode=image.mode)), flush=True)
        write_json(OUT / "zip_index.json", dict(members=[dict(name=v.filename, bytes=v.file_size,
            compressed_bytes=v.compress_size) for v in infos], annotation_rows=len(rows)))


def selection():
    """Freeze site-separated splits before any model results exist."""
    path = OUT / "selection.json"
    if path.exists(): return json.loads(path.read_text())
    rows = annotations()
    sites = sorted({r["name"].split("_")[0] for r in rows
                    if not r["name"].lower().startswith("tydemaniafortraining")})
    random.Random(42).shuffle(sites)
    site_splits = {site: ("test" if i < 8 else "validation" if i < 15 else "adaptation")
                   for i, site in enumerate(sites)}
    pools = defaultdict(list)
    for r in rows:
        site = r["name"].split("_")[0]
        if site not in site_splits: continue
        r = dict(r, site=site, image_group=r["name"].rsplit(".JPG_", 1)[0], group=target_group(r["label"]))
        pools[site_splits[site]].append(r)
    picked = []
    for split, pool in pools.items():
        random.Random(42).shuffle(pool)
        per_class, per_image = Counter(), Counter()
        for row in pool:
            if split == "adaptation":
                # Selection of adaptation inputs is independent of target labels.
                if per_image[row["image_group"]] or len([v for v in picked if v["split"] == split]) >= 64: continue
            elif row["group"] == 0 or per_class[row["group"]] >= (8 if split == "validation" else 16) or per_image[row["image_group"]]:
                continue
            picked.append(dict(row, split=split))
            per_class[row["group"]] += 1
            per_image[row["image_group"]] += 1
    result = dict(seed=42, source=DATASET, split_unit="site prefix in original photograph filename",
        site_splits=site_splits, rows=picked, groups=GROUPS, native_to_group=native_lookup(),
        selection="64 unlabeled adaptation patches; validation/test stratified up to 8/16 patches per common group, at most one patch per original photo",
        protocol="224x224 native patches; center-pixel proxy for supplied patch label (point transform unavailable); report patch-average prediction as sensitivity analysis",
        limitations=["Custom small pilot, not the official ReefNet benchmark or dense 95-class segmentation evaluation",
                     "Validation/test are class-stratified; accuracy/precision reflect that artificial prevalence",
                     "Filename prefixes are disjoint between adaptation, validation and test; geographic identity is not independently verified",
                     "RSG labels do not support a reliable bleaching benchmark (only two labeled bleached patches)"])
    write_json(path, result)
    print(json.dumps(dict(selected=Counter(r["split"] for r in picked), sites=Counter(site_splits.values()),
                         validation_groups=Counter(r["group"] for r in picked if r["split"]=="validation"),
                         test_groups=Counter(r["group"] for r in picked if r["split"]=="test"))), flush=True)
    return result


def prepare():
    from PIL import Image
    manifest = selection()
    file, archive = remote_zip()
    image_dir = OUT / "patches"
    image_dir.mkdir(exist_ok=True)
    with file, archive:
        entries = {Path(v.filename).name: v for v in archive.infolist() if not v.is_dir()}
        # Archive-offset order makes range reads cheaper and allows block reuse.
        pending = sorted(manifest["rows"], key=lambda r: entries[r["name"]].header_offset)
        for i, row in enumerate(pending):
            path = image_dir / row["name"]
            if not path.exists():
                data = archive.read(entries[row["name"]])
                with Image.open(io.BytesIO(data)) as image:
                    if image.size != (224,224) or image.mode != "RGB":
                        raise ValueError("RSG patch input protocol changed")
                path.write_bytes(data)
            row["path"] = str(path)
            row["sha256"] = sha256(path)
            if (i+1)%16 == 0: print(f"RSG selected patches {i+1}/{len(pending)}; range bytes {file.downloaded}",flush=True)
        manifest["range_download_bytes"] = file.downloaded
    for split in ["adaptation", "validation", "test"]:
        values = [dict(r) for r in manifest["rows"] if r["split"] == split]
        if split == "adaptation":
            # The actual training loader has no access to RSG ground truth.
            for r in values:
                r.pop("label"); r.pop("group")
        write_json(OUT / f"{split}.json", values)
    write_json(OUT / "data_manifest.json", {k:v for k,v in manifest.items() if k!="rows"})
    print(json.dumps(dict(status="prepared", patches=len(manifest["rows"]), range_bytes=file.downloaded)),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["inspect", "prepare"])
    args = parser.parse_args()
    {"inspect":inspect,"prepare":prepare}[args.action]()
