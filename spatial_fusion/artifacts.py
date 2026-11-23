"""Make our result recipes portable across NumPy/PyTorch versions."""

import hashlib
from pathlib import Path

import numpy as np
from numpy.core.multiarray import scalar
import torch


def make_portable(path):
    """Convert NumPy scalar metadata only; retain every saved tensor and number."""
    path = Path(path)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    allowed = [(scalar, 'numpy.core.multiarray.scalar'),
               (scalar, 'numpy._core.multiarray.scalar'), np.dtype,
               type(np.dtype('float64'))]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    converted = 0

    def plain(value):
        nonlocal converted
        if isinstance(value, np.generic):
            converted += 1
            return value.item()
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [plain(item) for item in value]
        if isinstance(value, tuple):
            return tuple(plain(item) for item in value)
        return value

    checkpoint = plain(checkpoint)
    if converted:
        checkpoint['artifact_serialization'] = dict(numpy_scalars_converted=converted,
            original_recipe_sha256=original_hash,
            policy='Convert scalar metadata to Python builtins; numerical values and tensors unchanged')
        temporary = path.with_suffix('.portable.tmp.pt')
        torch.save(checkpoint, temporary)
        torch.load(temporary, map_location='cpu', weights_only=True)
        temporary.replace(path)
    return converted
