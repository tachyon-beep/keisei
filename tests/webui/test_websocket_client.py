#!/usr/bin/env python3
"""
Simple WebSocket client to test what messages are being sent by the WebUI.
"""
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8765"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket server!")
            
            # Listen for messages
            message_count = 0
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_count += 1
                    
                    print(f"\n=== MESSAGE {message_count} ===")
                    print(f"Type: {data.get('type', 'UNKNOWN')}")
                    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
                    
                    if 'data' in data:
                        data_keys = list(data['data'].keys()) if isinstance(data['data'], dict) else 'N/A'
                        print(f"Data keys: {data_keys}")
                        
                        # Check for advanced visualization data
                        if data.get('type') == 'metrics_update':
                            metrics_data = data['data']
                            print(f"  🧬 Has entropy: {'entropy' in metrics_data}")
                            print(f"  🎯 Has skill_metrics: {'skill_metrics' in metrics_data}")  
                            print(f"  📈 Has gradient_norms: {'gradient_norms' in metrics_data}")
                            print(f"  🧠 Has policy_confidence: {'policy_confidence' in metrics_data}")
                            print(f"  ⚖️ Has value_estimate: {'value_estimate' in metrics_data}")
                            
                            if 'entropy' in metrics_data:
                                print(f"    Entropy value: {metrics_data['entropy']}")
                            if 'skill_metrics' in metrics_data and metrics_data['skill_metrics']:
                                print(f"    Skills: {list(metrics_data['skill_metrics'].keys())}")
                        
                        elif data.get('type') == 'progress_update':
                            progress_data = data['data']
                            print(f"  Steps: {progress_data.get('global_timestep', 'N/A')}")
                            print(f"  Episodes: {progress_data.get('total_episodes', 'N/A')}")
                            print(f"  Speed: {progress_data.get('speed', 'N/A')}")
                            
                        elif data.get('type') == 'board_update':
                            board_data = data['data']
                            print(f"  Board: {len(board_data.get('board', [])) if board_data.get('board') else 0}x{len(board_data.get('board', [[]]))} ")
                            print(f"  Current player: {board_data.get('current_player', 'N/A')}")
                    
                    if message_count >= 20:  # Stop after 20 messages
                        print("\n🛑 Received 20 messages, stopping...")
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON: {e}")
                    print(f"Raw message: {message}")
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    
    except ConnectionRefusedError:
        print("❌ Connection refused - WebSocket server not running on port 8765")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())