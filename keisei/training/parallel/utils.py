import gzip
import logging
from typing import Any, Dict

import numpy as np


def compress_array(array: np.ndarray) -> Dict[str, Any]:
    """Compress a numpy array with gzip for transmission."""
    try:
        array_bytes = array.tobytes()
        original_size = len(array_bytes)
        compressed_bytes = gzip.compress(array_bytes, compresslevel=6)
        compressed_size = len(compressed_bytes)
        compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else 1.0
        )
        return {
            "data": compressed_bytes,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "compressed": True,
            "compression_ratio": compression_ratio,
            "original_size": original_size,
            "compressed_size": compressed_size,
        }
    except Exception:
        # Fallback to uncompressed representation while preserving size metadata
        return {
            "data": array,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "compressed": False,
            "compression_ratio": 1.0,
            "original_size": array.nbytes,
            "compressed_size": array.nbytes,
        }


def decompress_array(data: Dict[str, Any]) -> np.ndarray:
    """Decompress array data produced by :func:`compress_array`."""
    try:
        if data.get("compressed", False):
            compressed_bytes = data["data"]
            decompressed_bytes = gzip.decompress(compressed_bytes)
            dtype = np.dtype(data["dtype"])
            shape = data["shape"]
            return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
        return data["data"]
    except Exception:
        logging.error("Decompression failed for array with shape=%s", data.get("shape"), exc_info=True)
        raise
