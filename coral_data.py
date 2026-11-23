"""Versioned Coralscapes source labels and the shared four-class health task."""

import json
from pathlib import Path

import numpy as np

CLASSES = ["Background", "Healthy", "Bleached", "Dead"]
CORAL_IDS = {1: [6, 17, 22, 25, 28, 31, 34, 36],
             2: [4, 16, 19, 33], 3: [3, 20, 23, 32, 37]}
IGNORE = 255
METADATA = Path(__file__).resolve().parent / "coralscapes_metadata"
LABEL_MAP = np.full(256, IGNORE, dtype=np.uint8)
LABEL_MAP[1:40] = 0
LABEL_MAP[[21, 27]] = IGNORE
for target, source in CORAL_IDS.items():
    LABEL_MAP[source] = target


def dataset_config(version="V1"):
    """Return a self-contained policy saved in each checkpoint and metrics file."""
    if version == "V1":
        names = json.loads((METADATA / "id2label.json").read_text())
        legacy = np.zeros(256, dtype=np.uint8)
        legacy[1:40] = np.arange(1, 40)
        settings = {"id": "EPFL-ECEO/coralscapes", "revision": None,
                    "directory": "coralscapes", "num_source_classes": 39,
                    "expected_split_sizes": {"train": 1517, "validation": 166, "test": 392}}
    elif version == "V2":
        metadata = json.loads((METADATA / "v2.json").read_text())
        names = metadata["id2label"]
        legacy = np.zeros(256, dtype=np.uint8)
        for source, target in metadata["fine_to_legacy_id"].items():
            legacy[int(source)] = target
        settings = {"id": metadata["dataset_id"], "revision": metadata["revision"],
                    "directory": "coralscapes_v2_" + metadata["revision"][:12],
                    "num_source_classes": 95, "mapping_source": metadata["mapping_source"],
                    "expected_split_sizes": {"train": 1790, "validation": 170, "test": 473}}
    else:
        raise ValueError("Annotation version must be V1 or V2.")
    return {**settings, "annotation_version": version, "configuration": "default",
            "source_labels": names, "source_to_legacy_id": legacy.tolist(),
            "label_map": LABEL_MAP[legacy].tolist(),
            "health_policy": "official legacy mapping, then four-class collapse; unlabeled/millepora/turbinaria ignored",
            "dead_category_taxonomy": "legacy Coralscapes IDs (V2 fine classes grouped when applicable)"}


def data_directory(root, settings):
    return Path(root) / "data" / settings["directory"]


def checkpoint_dataset(directory):
    """Read provenance before choosing annotation files, including old V1 checkpoints."""
    import torch
    state = torch.load(Path(directory) / "head.pt", map_location="cpu", weights_only=True)
    return state.get("dataset_config") or dataset_config("V1")
