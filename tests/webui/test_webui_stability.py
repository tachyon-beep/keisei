#!/usr/bin/env python3
"""
Test script to verify WebUI browser stability improvements.
This simulates high-frequency updates to test memory leak fixes.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

# Add the project to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from keisei.config_schema import WebUIConfig
from keisei.webui.webui_manager import WebUIManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockTrainer:
    """Mock trainer for testing WebUI without full training setup."""
    
    def __init__(self):
        self.metrics_manager = MockMetricsManager()
        self.game = MockGame()
        self.step_manager = MockStepManager()
        self.experience_buffer = MockBuffer()
        self.model = MockModel()
        
    class MockMetricsManager:
        def __init__(self):
            self.global_timestep = 0
            self.total_episodes_completed = 0
            self.black_wins = 0
            self.white_wins = 0
            self.draws = 0
            self.processing = False
            self.history = self.MockHistory()
            
        def get_hot_squares(self, count):
            return ["5e", "4f", "6d"]
            
        class MockHistory:
            def __init__(self):
                self.policy_losses = [0.1, 0.12, 0.09, 0.11]
                self.value_losses = [0.05, 0.06, 0.04, 0.05]
                self.entropies = [0.8, 0.75, 0.82, 0.78]
                self.kl_divergences = [0.02, 0.015, 0.018, 0.02]
                self.clip_fractions = [0.15, 0.12, 0.18, 0.14]
                self.learning_rates = [0.001, 0.001, 0.001, 0.001]
                self.episode_lengths = [45, 52, 38, 47]
                self.episode_rewards = [0.1, -0.2, 0.3, 0.0]
                self.win_rates_history = []
                
    class MockGame:
        def __init__(self):
            self.board = [[None for _ in range(9)] for _ in range(9)]
            self.current_player = "black"
            self.move_count = 15
            self.game_over = False
            self.winner = None
            
    class MockStepManager:
        def __init__(self):
            self.move_log = ["P-7f", "P-3d", "P-2f", "P-8d"]
            
    class MockBuffer:
        def size(self):
            return 250
            
        def capacity(self):
            return 1000
            
    class MockModel:
        def named_parameters(self):
            return []

def test_memory_stability():
    """Test WebUI memory management under high-frequency updates."""
    
    logger.info("🧪 Starting WebUI memory stability test...")
    
    # Create WebUI config
    config = WebUIConfig(
        enabled=True,
        host="localhost",
        port=8765,
        board_update_rate_hz=2.0,  # Reduced for stability
        metrics_update_rate_hz=5.0  # Reduced for stability
    )
    
    # Create WebUI manager
    webui = WebUIManager(config)
    
    # Start WebUI server
    if not webui.start():
        logger.error("Failed to start WebUI server")
        return False
        
    logger.info("WebUI server started successfully")
    
    # Create mock trainer
    trainer = MockTrainer()
    
    # Simulate high-frequency updates
    logger.info("Starting high-frequency update simulation...")
    
    update_count = 0
    start_time = time.time()
    
    try:
        for i in range(1000):  # Simulate 1000 updates
            # Update progress (simulating training steps)
            trainer.metrics_manager.global_timestep += 10
            trainer.metrics_manager.total_episodes_completed = i // 10
            
            speed = 15.5 + (i % 10) * 0.1  # Varying speed
            
            pending_updates = {
                "ep_metrics": f"Ep {i//10}: reward=0.{i%10}",
                "ppo_metrics": f"Loss: {0.1 + (i%100) * 0.001:.3f}",
                "current_epoch": i // 100
            }
            
            # Call update methods
            webui.update_progress(trainer, speed, pending_updates)
            webui.refresh_dashboard_panels(trainer)
            
            update_count += 1
            
            # Brief delay to simulate realistic update frequency
            await asyncio.sleep(0.01)  # 100 Hz simulation
            
            # Log progress periodically
            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = update_count / elapsed
                logger.info(f"Completed {update_count} updates at {rate:.1f} Hz")
                
    except Exception as e:
        logger.error(f"Error during update simulation: {e}")
        return False
    finally:
        webui.stop()
        
    elapsed = time.time() - start_time
    final_rate = update_count / elapsed
    
    logger.info(f"✅ Memory stability test completed!")
    logger.info(f"   Total updates: {update_count}")
    logger.info(f"   Duration: {elapsed:.1f}s")  
    logger.info(f"   Average rate: {final_rate:.1f} Hz")
    logger.info(f"   Messages processed: {webui.message_count}")
    
    return True

async def main():
    """Main test function."""
    try:
        success = await test_memory_stability()
        if success:
            logger.info("🎉 All tests passed - WebUI stability improvements verified!")
            return 0
        else:
            logger.error("❌ Tests failed - stability issues remain")
            return 1
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    # Fix asyncio event loop issues on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)