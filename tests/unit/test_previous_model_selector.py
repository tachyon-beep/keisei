"""Unit tests for keisei.training.previous_model_selector: PreviousModelSelector."""

from pathlib import Path

from keisei.training.previous_model_selector import PreviousModelSelector


class TestPreviousModelSelector:
    """Tests for the PreviousModelSelector deque-backed checkpoint pool."""

    def test_empty_pool_returns_none(self):
        selector = PreviousModelSelector()
        assert selector.get_random_checkpoint() is None

    def test_add_checkpoint_with_string(self):
        selector = PreviousModelSelector()
        selector.add_checkpoint("/tmp/model.pt")
        result = selector.get_random_checkpoint()
        assert result == Path("/tmp/model.pt")

    def test_add_checkpoint_with_path(self):
        selector = PreviousModelSelector()
        selector.add_checkpoint(Path("/tmp/model.pt"))
        result = selector.get_random_checkpoint()
        assert result == Path("/tmp/model.pt")

    def test_get_random_returns_one_of_added(self):
        selector = PreviousModelSelector(pool_size=5)
        paths = [f"/tmp/model_{i}.pt" for i in range(3)]
        for p in paths:
            selector.add_checkpoint(p)
        result = selector.get_random_checkpoint()
        assert result in [Path(p) for p in paths]

    def test_pool_overflow_drops_oldest(self):
        selector = PreviousModelSelector(pool_size=2)
        selector.add_checkpoint("/tmp/a.pt")
        selector.add_checkpoint("/tmp/b.pt")
        selector.add_checkpoint("/tmp/c.pt")
        all_items = list(selector.get_all())
        assert Path("/tmp/a.pt") not in all_items
        assert Path("/tmp/b.pt") in all_items
        assert Path("/tmp/c.pt") in all_items

    def test_get_all_returns_items_in_order(self):
        selector = PreviousModelSelector(pool_size=5)
        selector.add_checkpoint("/tmp/first.pt")
        selector.add_checkpoint("/tmp/second.pt")
        selector.add_checkpoint("/tmp/third.pt")
        all_items = list(selector.get_all())
        assert all_items == [
            Path("/tmp/first.pt"),
            Path("/tmp/second.pt"),
            Path("/tmp/third.pt"),
        ]

    def test_pool_size_one(self):
        selector = PreviousModelSelector(pool_size=1)
        selector.add_checkpoint("/tmp/a.pt")
        selector.add_checkpoint("/tmp/b.pt")
        assert list(selector.get_all()) == [Path("/tmp/b.pt")]
        assert selector.get_random_checkpoint() == Path("/tmp/b.pt")

    def test_default_pool_size_is_five(self):
        selector = PreviousModelSelector()
        assert selector.pool_size == 5
