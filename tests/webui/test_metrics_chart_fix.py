#!/usr/bin/env python3
"""
Test script to validate the metrics chart jittering fix.
This script tests the WebUI metrics table functionality to ensure
charts update smoothly without jittering.
"""

import asyncio
import websockets
import json
import time
import random
from typing import Dict, List

class MetricsChartTester:
    """Test class to simulate metrics updates and validate chart behavior"""
    
    def __init__(self):
        self.test_metrics = [
            "policy_loss", "value_loss", "entropy", "kl_divergence", 
            "clip_fraction", "episode_length", "episode_reward",
            "learning_rate", "gradient_norm"
        ]
        self.update_count = 0
        
    def generate_metrics_table(self, step: int) -> List[Dict]:
        """Generate realistic metrics table data for testing"""
        metrics_table = []
        
        for metric_name in self.test_metrics:
            # Simulate realistic training metric values with trends
            base_value = self._get_base_value(metric_name, step)
            noise = random.uniform(-0.1, 0.1)
            current_value = base_value * (1 + noise)
            prev_value = base_value * (1 + random.uniform(-0.15, 0.15))
            avg_value = (current_value + prev_value + base_value) / 3
            
            metrics_table.append({
                "name": metric_name,
                "last": current_value,
                "previous": prev_value,
                "average_5": avg_value
            })
            
        return metrics_table
    
    def _get_base_value(self, metric_name: str, step: int) -> float:
        """Get realistic base values for different metrics with training progression"""
        progress = min(step / 1000.0, 1.0)  # Normalize to 0-1
        
        if metric_name == "policy_loss":
            return 2.0 * (1 - progress * 0.7)  # Decreases from 2.0 to 0.6
        elif metric_name == "value_loss":  
            return 1.5 * (1 - progress * 0.6)  # Decreases from 1.5 to 0.6
        elif metric_name == "entropy":
            return 3.0 * (1 - progress * 0.5)  # Decreases from 3.0 to 1.5
        elif metric_name == "kl_divergence":
            return 0.02 * (1 + progress)  # Slightly increases
        elif metric_name == "clip_fraction":
            return 0.1 + 0.1 * progress  # Increases from 0.1 to 0.2
        elif metric_name == "episode_length":
            return 50 + 100 * progress  # Increases from 50 to 150
        elif metric_name == "episode_reward":
            return -10 + 15 * progress  # Increases from -10 to 5
        elif metric_name == "learning_rate":
            return 0.001 * (1 - progress * 0.5)  # Decreases
        elif metric_name == "gradient_norm":
            return 1.0 + 0.5 * random.random()  # Random around 1.0
        else:
            return random.uniform(0.1, 2.0)

    async def send_test_updates(self, websocket, num_updates: int = 50, delay: float = 0.1):
        """Send a series of metrics updates to test chart behavior"""
        print(f"🧪 Starting metrics chart jittering test...")
        print(f"📊 Sending {num_updates} updates with {delay}s delay")
        print(f"🎯 Testing chart persistence and smooth updates")
        
        for step in range(num_updates):
            self.update_count += 1
            
            # Generate realistic metrics table
            metrics_table = self.generate_metrics_table(step)
            
            # Create test message
            message = {
                "type": "metrics_update",
                "data": {
                    "metrics_table": metrics_table,
                    "global_timestep": step * 100,
                    "learning_curves": self._generate_learning_curves(step),
                    "game_statistics": self._generate_game_stats(step),
                    "buffer_info": {
                        "buffer_size": min(step * 10, 5000),
                        "buffer_capacity": 5000
                    },
                    "model_info": {
                        "gradient_norm": 1.0 + 0.5 * random.random()
                    },
                    "processing": step % 10 == 0  # Show processing indicator occasionally
                }
            }
            
            # Send message
            await websocket.send(json.dumps(message))
            print(f"📤 Sent update #{step+1}/{num_updates} - Metrics: {len(metrics_table)}")
            
            # Add some variation - occasionally add/remove metrics
            if step == 20:
                # Add a new metric mid-test
                self.test_metrics.append("test_new_metric")
                print("➕ Added new metric: test_new_metric")
            elif step == 35:
                # Remove a metric to test cleanup
                if "test_new_metric" in self.test_metrics:
                    self.test_metrics.remove("test_new_metric") 
                    print("➖ Removed metric: test_new_metric")
            
            await asyncio.sleep(delay)
            
        print(f"✅ Test completed! Sent {num_updates} updates")
        print(f"🔍 Check WebUI for smooth chart updates without jittering")
        
    def _generate_learning_curves(self, step: int) -> Dict:
        """Generate learning curve data for advanced visualizations"""
        length = min(step + 1, 50)  # Keep reasonable length
        
        return {
            "policy_losses": [random.uniform(0.5, 2.0) for _ in range(length)],
            "value_losses": [random.uniform(0.3, 1.5) for _ in range(length)],
            "entropies": [random.uniform(1.0, 3.0) for _ in range(length)],
            "kl_divergences": [random.uniform(0.001, 0.05) for _ in range(length)],
            "clip_fractions": [random.uniform(0.05, 0.3) for _ in range(length)],
            "episode_lengths": [random.uniform(30, 200) for _ in range(length)],
            "episode_rewards": [random.uniform(-20, 10) for _ in range(length)]
        }
        
    def _generate_game_stats(self, step: int) -> Dict:
        """Generate game statistics"""
        return {
            "games_per_hour": 100 + step,
            "average_game_length": 75 + random.randint(-10, 20),
            "win_loss_draw_rates": [
                {
                    "black_win_rate": 0.4 + random.uniform(-0.1, 0.1),
                    "white_win_rate": 0.35 + random.uniform(-0.1, 0.1), 
                    "draw_rate": 0.25 + random.uniform(-0.1, 0.1)
                }
                for _ in range(min(step + 1, 20))
            ]
        }

async def run_chart_jittering_test():
    """Main test function"""
    tester = MetricsChartTester()
    
    # Connect to WebUI WebSocket
    uri = "ws://localhost:8765"
    
    try:
        print(f"🔌 Connecting to WebUI at {uri}")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebUI WebSocket")
            
            # Send connection confirmation
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "Chart jittering test client connected"
            }))
            
            # Run the test
            await tester.send_test_updates(websocket, num_updates=50, delay=0.2)
            
            print("\n" + "="*60)
            print("🎯 CHART JITTERING TEST VALIDATION")
            print("="*60)
            print("✅ If charts update smoothly: FIX SUCCESSFUL")
            print("❌ If charts still jitter/reset: FIX NEEDS WORK") 
            print("🔍 Check browser DevTools for:")
            print("  - No 'Failed to update existing mini chart' warnings")
            print("  - Chart instances being reused")
            print("  - Smooth line progression without resets")
            print("="*60)
            
    except ConnectionRefusedError:
        print("❌ Failed to connect to WebUI WebSocket")
        print("💡 Make sure WebUI is running:")
        print("   python train.py train --override webui.enabled=true")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("🧪 Metrics Chart Jittering Fix Validator")
    print("=" * 50)
    print("This test validates that mini charts update smoothly")
    print("without the jittering issue caused by DOM recreation.")
    print("")
    
    # Run the test
    success = asyncio.run(run_chart_jittering_test())
    
    if success:
        print("\n🎉 Test execution completed successfully!")
    else:
        print("\n💥 Test execution failed!")