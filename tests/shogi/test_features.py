"""
test_features.py: Unit tests for keisei/shogi/features.py
"""

import pytest

from keisei.shogi import features


def test_feature_spec_num_planes():
    assert features.FEATURE_SPECS["core46"].num_planes == 46
    assert features.FEATURE_SPECS["core46+all"].num_planes == 51


def test_feature_specs_contains_all_entries():
    expected = {"core46", "core46+all", "dummyfeats", "testfeats", "resumefeats"}
    assert set(features.FEATURE_SPECS.keys()) == expected


def test_feature_spec_names_match_keys():
    for key, spec in features.FEATURE_SPECS.items():
        assert spec.name == key


def test_dummy_specs_have_correct_planes():
    assert features.FEATURE_SPECS["dummyfeats"].num_planes == 46
    assert features.FEATURE_SPECS["testfeats"].num_planes == 46
    assert features.FEATURE_SPECS["resumefeats"].num_planes == 46
