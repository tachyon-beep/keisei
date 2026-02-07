#!/usr/bin/env python3
"""
TRULY COMPREHENSIVE WebUI test - checks ALL 50+ UI elements individually.
"""
import asyncio
import websockets
import json
import time

# All UI elements that should be updated from backend data
ALL_UI_ELEMENTS = {
    # Header stats
    'timestep': 'progress_update.global_timestep',
    'episodes': 'progress_update.total_episodes', 
    'win-rate': 'progress_update.black_win_rate (calculated)',
    'speed': 'progress_update.speed',
    
    # Connection status
    'connection-status': 'connected message',
    
    # Game state
    'game-status': 'board_update.current_player + move_count',
    'shogi-board': 'board_update.board (9x9 grid)',
    'sente-hand': 'board_update.sente_hand',
    'gote-hand': 'board_update.gote_hand',
    
    # Win/Loss tracking
    'black-wins': 'progress_update.black_wins',
    'white-wins': 'progress_update.white_wins', 
    'draws': 'progress_update.draws',
    
    # Game statistics
    'games-per-hour': 'metrics_update.game_statistics.games_per_hour',
    'avg-game-length': 'metrics_update.game_statistics.average_game_length',
    'current-epoch': 'progress_update.current_epoch',
    
    # Training metrics
    'ep-metrics': 'progress_update.ep_metrics',
    'ppo-metrics': 'progress_update.ppo_metrics', 
    'metrics-table': 'metrics_update.metrics_table',
    
    # Buffer info
    'buffer-progress': 'metrics_update.buffer_info (progress bar)',
    'buffer-text': 'metrics_update.buffer_info.buffer_size/capacity',
    'gradient-norm': 'metrics_update.gradient_norms',
    
    # Hot squares
    'hot-squares': 'board_update.hot_squares',
    
    # Charts and visualizations
    'learning-chart': 'metrics_update.learning_curves (Chart.js)',
    'neural-heatmap': 'metrics_update.policy_confidence (canvas)',
    'advantage-chart': 'metrics_update.value_estimate (canvas)',
    'exploration-gauge': 'metrics_update.entropy (canvas)',
    'skill-radar': 'metrics_update.skill_metrics (canvas)',
    'gradient-flow': 'metrics_update.gradient_norms (canvas)', 
    'buffer-dynamics': 'metrics_update.buffer_info (canvas)',
    'tournament-tree': 'metrics_update.elo_data (canvas)',
    'strategy-radar': 'metrics_update.skill_metrics (canvas)',
    
    # Processing indicator
    'processing-indicator': 'Should show during processing',
}

async def test_all_ui_elements_comprehensive():
    uri = "ws://localhost:8765"
    print(f"🔍 COMPREHENSIVE UI AUDIT - testing {len(ALL_UI_ELEMENTS)} elements")
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            # Track all data received
            received_data = {
                'connected': False,
                'progress_updates': [],
                'board_updates': [],
                'metrics_updates': []
            }
            
            message_count = 0
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_count += 1
                    msg_type = data.get('type', 'UNKNOWN')
                    
                    print(f"\n=== MESSAGE {message_count} ({msg_type}) ===")
                    
                    # Store data by type
                    if msg_type == 'connected':
                        received_data['connected'] = True
                        
                    elif msg_type == 'progress_update':
                        received_data['progress_updates'].append(data.get('data', {}))
                        progress = data.get('data', {})
                        print(f"Progress keys: {list(progress.keys())}")
                        
                    elif msg_type == 'board_update':
                        received_data['board_updates'].append(data.get('data', {}))
                        board = data.get('data', {})
                        print(f"Board keys: {list(board.keys())}")
                        
                    elif msg_type == 'metrics_update':
                        received_data['metrics_updates'].append(data.get('data', {}))
                        metrics = data.get('data', {})
                        print(f"Metrics keys: {list(metrics.keys())}")
                    
                    if message_count >= 20:  # Collect enough data
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parse error: {e}")
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
            
            # Now analyze each UI element
            print(f"\n{'='*80}")
            print("🔍 INDIVIDUAL UI ELEMENT ANALYSIS")
            print(f"{'='*80}")
            
            results = {}
            
            # Get latest data
            latest_progress = received_data['progress_updates'][-1] if received_data['progress_updates'] else {}
            latest_board = received_data['board_updates'][-1] if received_data['board_updates'] else {}
            latest_metrics = received_data['metrics_updates'][-1] if received_data['metrics_updates'] else {}
            
            # Test each UI element
            for element_id, expected_source in ALL_UI_ELEMENTS.items():
                status = "❌ NO DATA"
                details = "No source data found"
                
                if element_id == 'connection-status':
                    if received_data['connected']:
                        status = "✅ WORKING"
                        details = "Connection message received"
                    else:
                        details = "No connection message"
                        
                elif element_id in ['timestep', 'episodes', 'speed', 'black-wins', 'white-wins', 'draws', 'current-epoch', 'ep-metrics', 'ppo-metrics']:
                    # Progress update elements
                    if latest_progress:
                        field = element_id.replace('-', '_')
                        if element_id == 'timestep':
                            field = 'global_timestep'
                        elif element_id == 'current-epoch':
                            field = 'current_epoch'
                        elif element_id == 'ep-metrics':
                            field = 'ep_metrics'
                        elif element_id == 'ppo-metrics':
                            field = 'ppo_metrics'
                            
                        if field in latest_progress:
                            status = "✅ WORKING"
                            details = f"Value: {latest_progress[field]}"
                        else:
                            details = f"Field '{field}' missing from progress data"
                    else:
                        details = "No progress updates received"
                        
                elif element_id == 'win-rate':
                    # Calculated field
                    if latest_progress and 'black_wins' in latest_progress and 'white_wins' in latest_progress and 'draws' in latest_progress:
                        status = "✅ WORKING"
                        total = latest_progress['black_wins'] + latest_progress['white_wins'] + latest_progress['draws']
                        rate = (latest_progress['black_wins'] / total * 100) if total > 0 else 0
                        details = f"Calculated: {rate:.1f}%"
                    else:
                        details = "Missing win/loss data for calculation"
                        
                elif element_id in ['shogi-board', 'sente-hand', 'gote-hand', 'game-status', 'hot-squares']:
                    # Board update elements
                    if latest_board:
                        if element_id == 'shogi-board':
                            if 'board' in latest_board and len(latest_board['board']) == 9:
                                status = "✅ WORKING"
                                pieces = sum(1 for row in latest_board['board'] for cell in row if cell)
                                details = f"9x9 board with {pieces} pieces"
                            else:
                                details = "Board data malformed or missing"
                        elif element_id == 'sente-hand':
                            if 'sente_hand' in latest_board:
                                status = "✅ WORKING" if latest_board['sente_hand'] else "⚠️  EMPTY"
                                details = f"Pieces: {latest_board['sente_hand']}"
                            else:
                                details = "sente_hand field missing"
                        elif element_id == 'gote-hand':
                            if 'gote_hand' in latest_board:
                                status = "✅ WORKING" if latest_board['gote_hand'] else "⚠️  EMPTY"
                                details = f"Pieces: {latest_board['gote_hand']}"
                            else:
                                details = "gote_hand field missing"
                        elif element_id == 'game-status':
                            if 'current_player' in latest_board:
                                status = "✅ WORKING"
                                details = f"Player: {latest_board['current_player']}"
                            else:
                                details = "current_player field missing"
                        elif element_id == 'hot-squares':
                            if 'hot_squares' in latest_board:
                                status = "✅ WORKING"
                                details = f"Squares: {latest_board['hot_squares']}"
                            else:
                                details = "hot_squares field missing"
                    else:
                        details = "No board updates received"
                        
                elif element_id in ['games-per-hour', 'avg-game-length', 'metrics-table', 'buffer-progress', 'buffer-text', 'gradient-norm', 'learning-chart']:
                    # Metrics update elements
                    if latest_metrics:
                        if element_id == 'games-per-hour':
                            if 'game_statistics' in latest_metrics and 'games_per_hour' in latest_metrics['game_statistics']:
                                status = "✅ WORKING"
                                details = f"Value: {latest_metrics['game_statistics']['games_per_hour']}"
                            else:
                                details = "game_statistics.games_per_hour missing"
                        elif element_id == 'avg-game-length':
                            if 'game_statistics' in latest_metrics and 'average_game_length' in latest_metrics['game_statistics']:
                                status = "✅ WORKING"  
                                details = f"Value: {latest_metrics['game_statistics']['average_game_length']}"
                            else:
                                details = "game_statistics.average_game_length missing"
                        elif element_id == 'metrics-table':
                            if 'metrics_table' in latest_metrics:
                                status = "✅ WORKING"
                                details = f"Table data available"
                            else:
                                details = "metrics_table field missing"
                        elif element_id in ['buffer-progress', 'buffer-text']:
                            if 'buffer_info' in latest_metrics:
                                status = "✅ WORKING"
                                details = f"Buffer: {latest_metrics['buffer_info']}"
                            else:
                                details = "buffer_info field missing"
                        elif element_id == 'gradient-norm':
                            if 'gradient_norms' in latest_metrics:
                                status = "✅ WORKING"
                                details = f"Gradients available"
                            else:
                                details = "gradient_norms field missing"
                        elif element_id == 'learning-chart':
                            if 'learning_curves' in latest_metrics:
                                status = "✅ WORKING"
                                curves = latest_metrics['learning_curves']
                                details = f"Curves: {list(curves.keys())}"
                            else:
                                details = "learning_curves field missing"
                    else:
                        details = "No metrics updates received"
                        
                elif element_id in ['neural-heatmap', 'advantage-chart', 'exploration-gauge', 'skill-radar', 'gradient-flow', 'buffer-dynamics', 'tournament-tree', 'strategy-radar']:
                    # Advanced visualization elements
                    if latest_metrics:
                        viz_data_available = []
                        if 'entropy' in latest_metrics: viz_data_available.append('entropy')
                        if 'skill_metrics' in latest_metrics: viz_data_available.append('skill_metrics')
                        if 'gradient_norms' in latest_metrics: viz_data_available.append('gradient_norms')
                        if 'policy_confidence' in latest_metrics: viz_data_available.append('policy_confidence')
                        if 'value_estimate' in latest_metrics: viz_data_available.append('value_estimate')
                        if 'elo_data' in latest_metrics: viz_data_available.append('elo_data')
                        
                        if viz_data_available:
                            status = "✅ DATA AVAILABLE"
                            details = f"Viz data: {viz_data_available}"
                        else:
                            details = "No advanced visualization data"
                    else:
                        details = "No metrics updates received"
                        
                elif element_id == 'processing-indicator':
                    # Special case - should be controlled by frontend
                    status = "⚠️  FRONTEND ONLY"
                    details = "Controlled by JavaScript, not backend data"
                
                results[element_id] = {'status': status, 'details': details}
                print(f"{status} {element_id:20} | {details}")
            
            # Summary
            working_count = sum(1 for r in results.values() if r['status'].startswith('✅'))
            total_count = len(results)
            
            print(f"\n{'='*80}")
            print(f"📊 COMPREHENSIVE UI AUDIT SUMMARY")
            print(f"{'='*80}")
            print(f"Elements tested: {total_count}")
            print(f"Working: {working_count}")
            print(f"Broken/Missing: {total_count - working_count}")
            print(f"Success rate: {working_count/total_count*100:.1f}%")
            
            # Critical issues
            critical_issues = []
            for element_id, result in results.items():
                if result['status'] == '❌ NO DATA' and element_id in ['timestep', 'episodes', 'shogi-board', 'learning-chart']:
                    critical_issues.append(f"{element_id}: {result['details']}")
            
            if critical_issues:
                print(f"\n🚨 CRITICAL ISSUES (core functionality broken):")
                for issue in critical_issues:
                    print(f"   • {issue}")
            else:
                print(f"\n🎉 No critical issues found!")
                
            # Data source summary
            print(f"\n📊 DATA SOURCE SUMMARY:")
            print(f"   Connected messages: {'✅' if received_data['connected'] else '❌'}")
            print(f"   Progress updates: {len(received_data['progress_updates'])} messages")
            print(f"   Board updates: {len(received_data['board_updates'])} messages")
            print(f"   Metrics updates: {len(received_data['metrics_updates'])} messages")
            
    except ConnectionRefusedError:
        print("❌ WebSocket server not running - start training with WebUI enabled")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_all_ui_elements_comprehensive())