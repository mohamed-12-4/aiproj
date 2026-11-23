"""Explicit common ecological groups; never invent dense labels from patch labels."""
import json
from .assets import ROOT

GROUPS = ["Unmapped", "Hard coral (condition unspecified)", "Dead hard coral", "Soft coral",
          "Algae", "Hard substrate", "Rubble", "Sediment", "Sponge", "Clam", "Human object"]
HARD = {"Acropora", "Porites", "Pocillopora", "Montipora", "Millepora", "Goniastrea", "Echinopora",
        "Favites", "Stylophora", "Lobophyllia", "Diploastrea", "Goniopora/Alveopora", "Seriatopora",
        "Galaxea", "Astreopora", "Tubastraea", "Plerogyra", "Bleached Hard Coral"}


def target_group(label):
    if label in HARD: return 1
    if label == "Dead Hard Coral": return 2
    if label in {"Other Soft Coral", "Xeniidae"}: return 3
    if label in {"Other Macroalgae", "Turf Algae", "CCA", "Tydemania", "Dictyota", "Halimeda"}: return 4
    return {"Hard substrate": 5, "Rubble": 6, "Sediment": 7, "Porifera": 8,
            "Tridacna": 9, "Human object": 10}.get(label, 0)


def native_lookup():
    labels = json.loads((ROOT / "coralscapes_metadata/v2.json").read_text())["id2label"]
    result = [0] * 96
    for key, name in labels.items():
        index = int(key)
        if name.startswith("algae covered"): group = 4
        elif name == "dead giant clam" or name == "giant clam": group = 9
        elif name.endswith(" dead"): group = 2
        elif name in {"leather coral alive", "nephtheidae alive", "soft coral (malacalcyonacea) alive",
                      "soft coral (malacalcyonacea) bleached", "xeniidae alive"}: group = 3
        elif name.endswith(" alive") or name.endswith(" bleached") or name in {"millepora","turbinaria"}: group = 1
        else: group = {"hard substrate": 5, "rubble": 6, "sand": 7, "sponge": 8,
                       "human-made structure": 10,"transect line":10}.get(name, 0)
        result[index] = group
    return result
