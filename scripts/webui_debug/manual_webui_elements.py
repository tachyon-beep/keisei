#!/usr/bin/env python3
"""
Comprehensive WebUI element test to identify all broken connections.
Tests data extraction, WebSocket transmission, and frontend element binding.
"""
import asyncio
import websockets
import json
import time

async def test_all_ui_elements():
    uri = "ws://localhost:8765"
    print(f"🧪 Testing WebUI elements - connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            # Track different message types and their data
            message_types = {}
            ui_elements_tested = {
                'connection_status': False,
                'board_updates': False,
                'progress_updates': False, 
                'metrics_updates': False,
                'learning_curves': False,
                'game_statistics': False,
                'buffer_info': False,
                'advanced_viz': False
            }
            
            message_count = 0
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_count += 1
                    msg_type = data.get('type', 'UNKNOWN')
                    
                    # Track message types
                    if msg_type not in message_types:
                        message_types[msg_type] = 0
                    message_types[msg_type] += 1
                    
                    print(f"\n=== MESSAGE {message_count} ({msg_type}) ===")
                    
                    if msg_type == 'connected':
                        ui_elements_tested['connection_status'] = True
                        print("✅ Connection status UI element: WORKING")
                        
                    elif msg_type == 'board_update':
                        ui_elements_tested['board_updates'] = True
                        board_data = data.get('data', {})
                        print("✅ Board updates: WORKING")
                        print(f"   Board size: {len(board_data.get('board', []))}x{len(board_data.get('board', [[]]))} ")
                        print(f"   Current player: {board_data.get('current_player', 'N/A')}")
                        print(f"   Game over: {board_data.get('game_over', False)}")
                        print(f"   Move count: {board_data.get('move_count', 0)}")
                        
                        # Test board pieces
                        board = board_data.get('board', [])
                        piece_count = 0
                        for row in board:
                            for cell in row:
                                if cell: piece_count += 1
                        print(f"   Pieces on board: {piece_count}")
                        
                        # Test hands
                        sente_pieces = sum(board_data.get('sente_hand', {}).values())
                        gote_pieces = sum(board_data.get('gote_hand', {}).values())
                        print(f"   Sente hand: {sente_pieces} pieces")
                        print(f"   Gote hand: {gote_pieces} pieces")
                        
                    elif msg_type == 'progress_update':
                        ui_elements_tested['progress_updates'] = True
                        progress_data = data.get('data', {})
                        print("✅ Progress updates: WORKING")
                        print(f"   Steps: {progress_data.get('global_timestep', 0)}")
                        print(f"   Episodes: {progress_data.get('total_episodes', 0)}")
                        print(f"   Speed: {progress_data.get('speed', 0):.2f}")
                        print(f"   Win rates: B{progress_data.get('black_wins', 0)} W{progress_data.get('white_wins', 0)} D{progress_data.get('draws', 0)}")
                        
                        # Check for missing progress fields
                        required_fields = ['global_timestep', 'speed', 'total_episodes', 'current_epoch']
                        missing_fields = [f for f in required_fields if f not in progress_data]
                        if missing_fields:
                            print(f"   ⚠️  Missing progress fields: {missing_fields}")
                            
                    elif msg_type == 'metrics_update':
                        ui_elements_tested['metrics_updates'] = True
                        metrics_data = data.get('data', {})
                        print("✅ Metrics updates: WORKING")
                        
                        # Test learning curves
                        if 'learning_curves' in metrics_data:
                            ui_elements_tested['learning_curves'] = True
                            curves = metrics_data['learning_curves']
                            print(f"   📊 Learning curves: {list(curves.keys())}")
                        else:
                            print("   ❌ Learning curves: MISSING")
                            
                        # Test game statistics  
                        if 'game_statistics' in metrics_data:
                            ui_elements_tested['game_statistics'] = True
                            stats = metrics_data['game_statistics']
                            print(f"   📈 Game statistics: {list(stats.keys())}")
                        else:
                            print("   ❌ Game statistics: MISSING")
                            
                        # Test buffer info
                        if 'buffer_info' in metrics_data:
                            ui_elements_tested['buffer_info'] = True
                            buffer = metrics_data['buffer_info']
                            print(f"   🔄 Buffer info: {list(buffer.keys())}")
                        else:
                            print("   ❌ Buffer info: MISSING")
                            
                        # Test advanced visualization data
                        viz_fields = ['entropy', 'skill_metrics', 'gradient_norms', 'policy_confidence', 'value_estimate']
                        found_viz = [f for f in viz_fields if f in metrics_data]
                        if found_viz:
                            ui_elements_tested['advanced_viz'] = True
                            print(f"   🎯 Advanced viz data: {found_viz}")
                        else:
                            print("   ❌ Advanced visualization data: MISSING")
                    
                    if message_count >= 15:  # Test enough messages
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse error: {e}")
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
            
            # Summary report
            print(f"\n{'='*50}")
            print("📋 WEBUI ELEMENT TEST SUMMARY")
            print(f"{'='*50}")
            print(f"Messages processed: {message_count}")
            print(f"Message types: {list(message_types.keys())}")
            
            print(f"\n🧩 UI ELEMENT STATUS:")
            for element, working in ui_elements_tested.items():
                status = "✅ WORKING" if working else "❌ BROKEN"
                print(f"   {element}: {status}")
                
            # Identify critical issues
            critical_issues = []
            if not ui_elements_tested['board_updates']:
                critical_issues.append("Board not updating - pieces won't show")
            if not ui_elements_tested['learning_curves']:
                critical_issues.append("Learning curves missing - chart will be empty")  
            if not ui_elements_tested['buffer_info']:
                critical_issues.append("Buffer progress missing - progress bars won't work")
            if not ui_elements_tested['advanced_viz']:
                critical_issues.append("Advanced visualizations missing - dashboard will be empty")
                
            if critical_issues:
                print(f"\n🚨 CRITICAL ISSUES FOUND:")
                for issue in critical_issues:
                    print(f"   • {issue}")
            else:
                print(f"\n🎉 ALL UI ELEMENTS WORKING CORRECTLY!")
                    
    except ConnectionRefusedError:
        print("❌ WebSocket server not running - start training with WebUI enabled")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_ui_elements())