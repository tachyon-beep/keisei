"""Unit tests for keisei.shogi.features: FeatureSpec and FEATURE_SPECS registry."""

from keisei.shogi.features import FEATURE_SPECS, FeatureSpec


class TestFeatureSpec:
    """Tests for the FeatureSpec dataclass."""

    def test_construction_and_attributes(self):
        spec = FeatureSpec("test", 32)
        assert spec.name == "test"
        assert spec.num_planes == 32

    def test_zero_planes(self):
        spec = FeatureSpec("empty", 0)
        assert spec.num_planes == 0


class TestFeatureSpecsRegistry:
    """Tests for the FEATURE_SPECS dictionary."""

    def test_registry_has_exactly_five_entries(self):
        assert len(FEATURE_SPECS) == 5

    def test_registry_keys(self):
        expected_keys = {"core46", "core46+all", "dummyfeats", "testfeats", "resumefeats"}
        assert set(FEATURE_SPECS.keys()) == expected_keys

    def test_core46_spec(self):
        spec = FEATURE_SPECS["core46"]
        assert spec.name == "core46"
        assert spec.num_planes == 46

    def test_core46_all_spec(self):
        spec = FEATURE_SPECS["core46+all"]
        assert spec.name == "core46+all"
        assert spec.num_planes == 51

    def test_dummyfeats_spec(self):
        spec = FEATURE_SPECS["dummyfeats"]
        assert spec.name == "dummyfeats"
        assert spec.num_planes == 46

    def test_testfeats_spec(self):
        spec = FEATURE_SPECS["testfeats"]
        assert spec.name == "testfeats"
        assert spec.num_planes == 46

    def test_resumefeats_spec(self):
        spec = FEATURE_SPECS["resumefeats"]
        assert spec.name == "resumefeats"
        assert spec.num_planes == 46

    def test_all_values_are_featurespec_instances(self):
        for key, spec in FEATURE_SPECS.items():
            assert isinstance(spec, FeatureSpec), f"{key} is not a FeatureSpec"
