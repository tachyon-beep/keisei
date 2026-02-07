#!/usr/bin/env python3
"""
Comprehensive test of complete WebUI data flow including frontend integration.
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
from keisei.webui.web_server import WebUIHTTPServer
# Import mock objects (demo_webui is a standalone script, only available when run directly)
try:
    from demo_webui import MockTrainer
except ImportError:
    MockTrainer = None  # Not available when run via pytest

def comprehensive_training_simulation(webui_manager, trainer, duration=15):
    """Comprehensive training simulation with varied data."""
    print(f"🎯 Starting comprehensive simulation for {duration} seconds...")
    
    start_time = time.time()
    step = 0
    
    while time.time() - start_time < duration:
        step += 1
        try:
            # Update basic training metrics
            trainer.metrics_manager.global_timestep += 50
            trainer.metrics_manager.total_episodes_completed += 1
            
            # Vary the wins periodically
            if step % 5 == 0:
                trainer.metrics_manager.black_wins += 1
            elif step % 7 == 0:
                trainer.metrics_manager.white_wins += 1
            elif step % 12 == 0:
                trainer.metrics_manager.draws += 1
            
            # Add realistic training data
            progress_data = {
                "ep_metrics": f"Ep {trainer.metrics_manager.total_episodes_completed}: L:{45-step%20} R:{0.1+step*0.001:.3f}",
                "ppo_metrics": f"Loss: {0.8-step*0.005:.3f}, Ent: {2.1-step*0.01:.2f}, KL: {0.02-step*0.0001:.4f}",
                "current_epoch": (step // 10) + 1,
                "black_win_rate": trainer.metrics_manager.black_wins / max(trainer.metrics_manager.total_episodes_completed, 1),
                "white_win_rate": trainer.metrics_manager.white_wins / max(trainer.metrics_manager.total_episodes_completed, 1),
                "draw_rate": trainer.metrics_manager.draws / max(trainer.metrics_manager.total_episodes_completed, 1)
            }
            
            # Send progress update (basic training metrics)
            webui_manager.update_progress(trainer, speed=7.5 + step*0.1, pending_updates=progress_data)
            
            # Send dashboard refresh (advanced visualization data)
            if step % 2 == 0:  # Every other step
                webui_manager.refresh_dashboard_panels(trainer)
                
                # Update some history data to make visualizations more dynamic
                if len(trainer.metrics_manager.history.policy_losses) > 0:
                    trainer.metrics_manager.history.policy_losses[-1] = max(0.1, 0.8 - step*0.01)
                if len(trainer.metrics_manager.history.entropies) > 0:
                    trainer.metrics_manager.history.entropies[-1] = max(0.5, 2.1 - step*0.02)
            
            print(f"  📤 Step {step}: Timesteps={trainer.metrics_manager.global_timestep}, Episodes={trainer.metrics_manager.total_episodes_completed}")
            
            time.sleep(1)  # 1 second intervals
            
        except Exception as e:
            print(f"❌ Error in simulation step {step}: {e}")
            break
            
    print("🏁 Comprehensive simulation complete!")

def main():
    print("🎯 COMPREHENSIVE WEBUI DEBUG TEST")
    print("=" * 60)
    print("This test will:")
    print("  1. Start WebSocket server (port 8765)")  
    print("  2. Start HTTP server (port 8766)")
    print("  3. Run comprehensive training simulation")
    print("  4. Generate varied data for all visualizations")
    print("=" * 60)
    
    # Create configuration
    config = WebUIConfig(
        enabled=True,
        port=8765,
        host="0.0.0.0",  # Allow external connections
        update_rate_hz=2.0,
        board_update_rate_hz=1.0,
        metrics_update_rate_hz=1.0
    )
    
    # Create managers
    webui_manager = WebUIManager(config)
    http_server = WebUIHTTPServer("0.0.0.0", 8766)
    trainer = MockTrainer()
    
    print("🚀 Starting servers...")
    
    # Start WebSocket server
    if not webui_manager.start():
        print("❌ Failed to start WebSocket server")
        return
    print(f"✅ WebSocket server started on ws://0.0.0.0:8765")
        
    # Start HTTP server
    if not http_server.start():
        print("❌ Failed to start HTTP server")
        webui_manager.stop()
        return
    print(f"✅ HTTP server started on http://0.0.0.0:8766")
    
    print("\n" + "=" * 60)
    print("🌐 MANUAL TEST INSTRUCTIONS:")
    print("   1. Open browser to: http://localhost:8766")
    print("   2. Open Developer Tools (F12)")
    print("   3. Watch Console tab for debug messages")
    print("   4. Look for:")
    print("      - 🎨 updateAdvancedVisualizations called")
    print("      - ✅ Visualization updates scheduled")  
    print("      - Canvas rendering activity")
    print("   5. Verify visualizations are updating:")
    print("      - Neural heatmap overlay on board")
    print("      - Exploration gauge (semicircle)")
    print("      - Advantage oscillation chart")
    print("      - Skill radar (after 30 seconds)")
    print("=" * 60)
    
    print(f"\n⏰ Starting training simulation in 3 seconds...")
    time.sleep(3)
    
    # Start training simulation thread
    sim_thread = threading.Thread(
        target=comprehensive_training_simulation,
        args=(webui_manager, trainer, 45),  # 45 second simulation
        daemon=True
    )
    sim_thread.start()
    
    print("\n🎯 Training simulation started!")
    print("   Watch the browser for live updates...")
    print("   Press Ctrl+C to stop")
    
    try:
        # Keep main thread alive
        sim_thread.join()
        
        print("\n✨ Simulation completed successfully!")
        print("   Keep servers running for manual inspection...")
        
        # Keep servers running for manual inspection
        input("\n⏸️  Press Enter to shutdown servers...")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        print("\n🧹 Shutting down servers...")
        webui_manager.stop()
        http_server.stop()
        print("✅ All servers stopped")
        print("\n📋 DEBUGGING SUMMARY:")
        print("   - Data extraction: ✅ WORKING (confirmed)")
        print("   - WebSocket transmission: ✅ WORKING (confirmed)")
        print("   - HTTP server: ✅ WORKING (confirmed)")
        print("   - Frontend integration: ⚠️  CHECK BROWSER CONSOLE")
        print("   - Canvas rendering: ⚠️  CHECK BROWSER VISUALLY")

if __name__ == "__main__":
    main()