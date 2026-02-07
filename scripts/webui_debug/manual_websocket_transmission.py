#!/usr/bin/env python3
"""
Test WebSocket transmission of advanced visualization data.
"""

import asyncio
import json
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from keisei.config_schema import WebUIConfig
from keisei.webui.webui_manager import WebUIManager

# Import mock objects (demo_webui is a standalone script, only available when run directly)
try:
    from demo_webui import MockTrainer
except ImportError:
    MockTrainer = None  # Not available when run via pytest

async def websocket_client_test():
    """Test client to verify WebSocket messages."""
    try:
        import websockets
    except ImportError:
        print("❌ websockets package not available")
        return
        
    uri = "ws://localhost:8765"
    
    try:
        print(f"🔌 Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected")
            
            message_count = 0
            start_time = time.time()
            
            while time.time() - start_time < 10:  # Listen for 10 seconds
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    print(f"\n📨 Message {message_count}: {data['type']}")
                    
                    if data['type'] == 'metrics_update':
                        msg_data = data.get('data', {})
                        print(f"  🎨 Advanced vis fields present:")
                        
                        advanced_fields = [
                            'entropy', 'value_estimate', 'policy_confidence', 
                            'skill_metrics', 'gradient_norms'
                        ]
                        
                        for field in advanced_fields:
                            if field in msg_data:
                                value = msg_data[field]
                                if isinstance(value, list):
                                    print(f"    ✅ {field}: list[{len(value)}]")
                                elif isinstance(value, dict):
                                    print(f"    ✅ {field}: dict[{list(value.keys())}]")
                                else:
                                    print(f"    ✅ {field}: {value}")
                            else:
                                print(f"    ❌ {field}: MISSING")
                        
                        # Check buffer info
                        if 'buffer_info' in msg_data:
                            buffer_info = msg_data['buffer_info']
                            if 'quality_distribution' in buffer_info:
                                quality = buffer_info['quality_distribution']
                                print(f"    ✅ buffer quality_distribution: list[{len(quality)}]")
                            else:
                                print(f"    ❌ buffer quality_distribution: MISSING")
                    
                except asyncio.TimeoutError:
                    print("⏰ No message received in 2 seconds")
                    continue
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    break
                    
            print(f"\n📊 Total messages received: {message_count}")
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")

def simulate_training_with_advanced_data(webui_manager, trainer):
    """Simulate training with focus on advanced data."""
    print("🎯 Starting advanced data simulation...")
    
    for step in range(3):  # Just 3 steps for testing
        try:
            # Update basic progress
            trainer.metrics_manager.global_timestep += 100
            trainer.metrics_manager.total_episodes_completed += 2
            
            progress_data = {
                "ep_metrics": f"Ep {trainer.metrics_manager.total_episodes_completed}: L:42 R:0.123",
                "ppo_metrics": f"Loss: 0.8, Ent: 2.1",
                "current_epoch": step + 1
            }
            
            print(f"  📤 Sending progress_update (step {step+1})...")
            webui_manager.update_progress(trainer, speed=8.5, pending_updates=progress_data)
            
            time.sleep(1)
            
            print(f"  📤 Sending dashboard refresh (advanced data) (step {step+1})...")
            webui_manager.refresh_dashboard_panels(trainer)
            
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error in simulation step {step}: {e}")
            break
            
    print("🏁 Advanced data simulation complete!")

def main():
    print("🧪 WebSocket Advanced Data Transmission Test")
    print("=" * 50)

    if MockTrainer is None:
        print("❌ demo_webui module not found. Run this script from the project root.")
        return

    # Create WebUI manager
    config = WebUIConfig(
        enabled=True,
        port=8765,
        host="localhost",
        update_rate_hz=1.0,
        board_update_rate_hz=0.5,
        metrics_update_rate_hz=0.5
    )
    
    webui_manager = WebUIManager(config)
    trainer = MockTrainer()
    
    # Start WebSocket server
    print("🚀 Starting WebSocket server...")
    if not webui_manager.start():
        print("❌ Failed to start WebSocket server")
        return
        
    print("✅ WebSocket server started")
    time.sleep(1)
    
    # Start training simulation in background thread
    sim_thread = threading.Thread(
        target=simulate_training_with_advanced_data, 
        args=(webui_manager, trainer),
        daemon=True
    )
    sim_thread.start()
    
    # Run the client test
    print("🎧 Starting WebSocket client test...")
    try:
        asyncio.run(websocket_client_test())
    except Exception as e:
        print(f"❌ Client test failed: {e}")
    finally:
        print("\n🧹 Cleaning up...")
        webui_manager.stop()
        print("✅ Test complete!")

if __name__ == "__main__":
    main()