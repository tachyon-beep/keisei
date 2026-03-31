"""
features.py: FeatureSpec metadata for Keisei Shogi observation tensors.

Observation generation is handled by shogi_game_io.py. This module provides
FeatureSpec entries that describe plane counts for each feature set, used by
model_manager.py to determine input shape.
"""

from typing import Any, Dict, Optional

import numpy as np


class FeatureSpec:
    """
    Describes a set of feature planes for Shogi observation tensors.

    The num_planes attribute is used by model_manager to determine the
    observation tensor shape. Observation generation itself is in shogi_game_io.py.
    """

    def __init__(self, name: str, num_planes: int):
        self.name = name
        self.num_planes = num_planes


# Feature set specifications (plane counts only)
CORE46_SPEC = FeatureSpec("core46", 46)
CORE46_ALL_SPEC = FeatureSpec("core46+all", 51)
DUMMY_FEATS_SPEC = FeatureSpec("dummyfeats", 46)
TEST_FEATS_SPEC = FeatureSpec("testfeats", 46)
RESUME_FEATS_SPEC = FeatureSpec("resumefeats", 46)

FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "core46": CORE46_SPEC,
    "core46+all": CORE46_ALL_SPEC,
    "dummyfeats": DUMMY_FEATS_SPEC,
    "testfeats": TEST_FEATS_SPEC,
    "resumefeats": RESUME_FEATS_SPEC,
}
