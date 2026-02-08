"""Unit tests for keisei.training.parallel.communication.WorkerCommunicator."""

import multiprocessing as mp
import queue
import time

import numpy as np
import pytest
import torch

from keisei.training.parallel.communication import WorkerCommunicator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def comm():
    """WorkerCommunicator with 2 workers and small queues for fast tests."""
    c = WorkerCommunicator(num_workers=2, max_queue_size=5, timeout=1.0)
    yield c
    # Best-effort cleanup — drain and close queues so child threads can exit
    for qlist in [c.experience_queues, c.model_queues, c.control_queues]:
        for q in qlist:
            try:
                while not q.empty():
                    q.get_nowait()
            except (queue.Empty, OSError):
                pass
            try:
                q.close()
            except OSError:
                pass


@pytest.fixture
def small_state_dict():
    """A tiny model state dict for transmission tests."""
    return {"weight": torch.randn(4, 3), "bias": torch.randn(4)}


# ---------------------------------------------------------------------------
# TestWorkerCommunicatorInit
# ---------------------------------------------------------------------------


class TestWorkerCommunicatorInit:
    def test_creates_correct_number_of_queues(self, comm):
        assert len(comm.experience_queues) == 2
        assert len(comm.model_queues) == 2
        assert len(comm.control_queues) == 2

    def test_queue_max_sizes(self, comm):
        # mp.Queue stores maxsize as _maxsize (implementation detail).
        # Verify behaviorally: experience queues accept max_queue_size items
        # but not more.
        for i in range(5):
            comm.experience_queues[0].put_nowait({"i": i})
        with pytest.raises(queue.Full):
            comm.experience_queues[0].put_nowait({"overflow": True})
        # model queues accept 10
        for i in range(10):
            comm.model_queues[0].put_nowait({"i": i})
        with pytest.raises(queue.Full):
            comm.model_queues[0].put_nowait({"overflow": True})

    def test_timeout_stored(self, comm):
        assert comm.timeout == 1.0


# ---------------------------------------------------------------------------
# TestSendModelWeights
# ---------------------------------------------------------------------------


class TestSendModelWeights:
    def test_all_workers_receive_weights(self, comm, small_state_dict):
        comm.send_model_weights(small_state_dict, compression_enabled=False)
        # Use get(timeout=...) rather than empty() to avoid race conditions
        for mq in comm.model_queues:
            item = mq.get(timeout=2)
            assert "model_data" in item

    def test_compressed_payload_structure(self, comm, small_state_dict):
        comm.send_model_weights(small_state_dict, compression_enabled=True)
        payload = comm.model_queues[0].get(timeout=2)
        assert "model_data" in payload
        assert "timestamp" in payload
        assert "compressed" in payload
        assert "compression_ratio" in payload
        assert payload["compressed"] is True

    def test_uncompressed_payload(self, comm, small_state_dict):
        comm.send_model_weights(small_state_dict, compression_enabled=False)
        payload = comm.model_queues[0].get(timeout=2)
        assert payload["compressed"] is False

    def test_queue_full_does_not_crash(self, comm, small_state_dict):
        """Fill queue to capacity, then send again — should not raise."""
        # model_queues have maxsize=10, fill them up
        for _ in range(10):
            for mq in comm.model_queues:
                try:
                    mq.put_nowait({"dummy": True})
                except queue.Full:
                    pass
        # This should not raise, just log a warning
        comm.send_model_weights(small_state_dict, compression_enabled=False)


# ---------------------------------------------------------------------------
# TestCollectExperiences
# ---------------------------------------------------------------------------


class TestCollectExperiences:
    def test_empty_queues_returns_empty_list(self, comm):
        result = comm.collect_experiences()
        assert result == []

    def test_collects_from_all_workers(self, comm):
        comm.experience_queues[0].put({"data": "from_worker_0"})
        comm.experience_queues[1].put({"data": "from_worker_1"})
        # Allow data to propagate through the mp.Queue internal pipe
        time.sleep(0.1)
        result = comm.collect_experiences()
        assert len(result) == 2
        worker_ids = {r[0] for r in result}
        assert worker_ids == {0, 1}

    def test_returns_worker_id_and_data_tuples(self, comm):
        comm.experience_queues[0].put({"obs": [1, 2, 3]})
        time.sleep(0.1)
        result = comm.collect_experiences()
        assert len(result) == 1
        worker_id, batch_data = result[0]
        assert worker_id == 0
        assert batch_data == {"obs": [1, 2, 3]}

    def test_drains_multiple_items_per_worker(self, comm):
        for i in range(3):
            comm.experience_queues[0].put({"batch": i})
        time.sleep(0.1)
        result = comm.collect_experiences()
        assert len(result) == 3
        assert all(wid == 0 for wid, _ in result)


# ---------------------------------------------------------------------------
# TestSendControlCommand
# ---------------------------------------------------------------------------


class TestSendControlCommand:
    def test_broadcast_to_all_workers(self, comm):
        comm.send_control_command("pause", data={"duration": 0.0})
        # Use get(timeout=...) to verify all workers received the command
        for cq in comm.control_queues:
            msg = cq.get(timeout=2)
            assert msg["command"] == "pause"

    def test_send_to_specific_workers(self, comm):
        comm.send_control_command("reset", worker_ids=[0])
        msg = comm.control_queues[0].get(timeout=2)
        assert msg["command"] == "reset"
        # Worker 1's queue should remain empty
        time.sleep(0.1)
        assert comm.control_queues[1].empty()

    def test_command_message_structure(self, comm):
        comm.send_control_command("stop", data={"reason": "test"})
        msg = comm.control_queues[0].get(timeout=2)
        assert msg["command"] == "stop"
        assert msg["data"] == {"reason": "test"}
        assert "timestamp" in msg

    def test_stop_command(self, comm):
        comm.send_control_command("stop")
        msg = comm.control_queues[0].get(timeout=2)
        assert msg["command"] == "stop"


# ---------------------------------------------------------------------------
# TestPrepareModelData
# ---------------------------------------------------------------------------


class TestPrepareModelData:
    def test_tensors_converted_to_numpy(self, comm, small_state_dict):
        result = comm._prepare_model_data(small_state_dict, compress=False)
        for key, val in result["model_data"].items():
            # The "data" field should be numpy, not a torch.Tensor
            assert isinstance(val["data"], np.ndarray)

    def test_compression_ratio_calculated(self, comm, small_state_dict):
        result = comm._prepare_model_data(small_state_dict, compress=True)
        assert result["compression_ratio"] > 0

    def test_size_tracking(self, comm, small_state_dict):
        result = comm._prepare_model_data(small_state_dict, compress=True)
        assert "total_original_size" in result
        assert "total_compressed_size" in result
        assert result["total_original_size"] > 0
        assert result["total_compressed_size"] > 0


# ---------------------------------------------------------------------------
# TestGetQueueInfo
# ---------------------------------------------------------------------------


class TestGetQueueInfo:
    def test_initial_sizes_all_zero(self, comm):
        info = comm.get_queue_info()
        assert info["experience_queue_sizes"] == [0, 0]
        assert info["model_queue_sizes"] == [0, 0]
        assert info["control_queue_sizes"] == [0, 0]

    def test_sizes_reflect_queue_state(self, comm):
        comm.experience_queues[0].put({"x": 1})
        comm.experience_queues[0].put({"x": 2})
        comm.control_queues[1].put({"cmd": "stop"})
        time.sleep(0.1)
        info = comm.get_queue_info()
        assert info["experience_queue_sizes"][0] == 2
        assert info["experience_queue_sizes"][1] == 0
        assert info["control_queue_sizes"][1] == 1


# ---------------------------------------------------------------------------
# TestCleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_closes_queues(self):
        """After cleanup, queues are closed and cannot accept new items."""
        c = WorkerCommunicator(num_workers=2, max_queue_size=5, timeout=1.0)
        c.experience_queues[0].put({"data": 1})
        time.sleep(0.1)
        c.cleanup()
        # After close(), putting should raise OSError
        with pytest.raises((ValueError, OSError)):
            c.experience_queues[0].put({"data": 1})

    def test_cleanup_does_not_raise(self):
        """Cleanup on a communicator with data in queues completes without error."""
        c = WorkerCommunicator(num_workers=1, max_queue_size=5, timeout=1.0)
        c.experience_queues[0].put({"data": 1})
        c.experience_queues[0].put({"data": 2})
        time.sleep(0.1)
        c.cleanup()  # Should not raise

    def test_cleanup_sends_stop_to_control_queues(self):
        """Verify cleanup sends stop by intercepting before drain."""
        from unittest.mock import patch

        c = WorkerCommunicator(num_workers=1, max_queue_size=5, timeout=1.0)
        # Capture whether send_control_command was called with "stop"
        with patch.object(c, "send_control_command", wraps=c.send_control_command) as mock_send:
            c.cleanup()
            mock_send.assert_called_once_with("stop")
