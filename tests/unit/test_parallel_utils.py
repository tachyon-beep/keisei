"""Unit tests for keisei.training.parallel.utils: compress_array / decompress_array."""

import numpy as np
import pytest

from keisei.training.parallel.utils import compress_array, decompress_array


class TestCompressDecompressRoundTrip:
    """Tests that compression followed by decompression preserves array data."""

    def test_float32_round_trip(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = decompress_array(compress_array(arr))
        np.testing.assert_array_equal(result, arr)

    def test_float64_round_trip(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = decompress_array(compress_array(arr))
        np.testing.assert_array_equal(result, arr)

    def test_int32_round_trip(self):
        arr = np.array([10, 20, 30], dtype=np.int32)
        result = decompress_array(compress_array(arr))
        np.testing.assert_array_equal(result, arr)

    def test_bool_round_trip(self):
        arr = np.array([True, False, True, False], dtype=bool)
        result = decompress_array(compress_array(arr))
        np.testing.assert_array_equal(result, arr)

    def test_2d_array_round_trip(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = decompress_array(compress_array(arr))
        np.testing.assert_array_equal(result, arr)
        assert result.shape == (3, 4)

    def test_3d_array_round_trip(self):
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        result = decompress_array(compress_array(arr))
        np.testing.assert_array_equal(result, arr)
        assert result.shape == (2, 3, 4)

    def test_1d_array_round_trip(self):
        arr = np.array([42], dtype=np.float32)
        result = decompress_array(compress_array(arr))
        np.testing.assert_array_equal(result, arr)


class TestCompressArray:
    """Tests for compress_array output structure."""

    def test_compressed_result_has_expected_keys(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = compress_array(arr)
        assert result["compressed"] is True
        assert result["shape"] == (3,)
        assert result["dtype"] == "float32"
        assert "data" in result
        assert "compression_ratio" in result
        assert "original_size" in result
        assert "compressed_size" in result

    def test_compression_ratio_positive(self):
        arr = np.zeros(1000, dtype=np.float32)
        result = compress_array(arr)
        assert result["compression_ratio"] > 1.0  # Zeros compress well


class TestDecompressArray:
    """Tests for decompress_array with uncompressed fallback data."""

    def test_decompress_uncompressed_data(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        data = {
            "data": arr,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "compressed": False,
        }
        result = decompress_array(data)
        np.testing.assert_array_equal(result, arr)

    def test_empty_array_round_trip(self):
        arr = np.array([], dtype=np.float32)
        result = decompress_array(compress_array(arr))
        np.testing.assert_array_equal(result, arr)
