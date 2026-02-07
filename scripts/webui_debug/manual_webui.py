#!/usr/bin/env python3
"""
Test script for WebUI functionality.
"""

import sys
import time
from pathlib import Path

# Add keisei to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from keisei.config_schema import WebUIConfig
    from keisei.webui.webui_manager import WebUIManager
    from keisei.webui.web_server import WebUIHTTPServer
    
    print("✓ WebUI imports successful")
except ImportError as e:
    print(f"✗ WebUI import failed: {e}")
    print("Try installing websockets: pip install websockets")
    sys.exit(1)


def test_webui():
    """Test WebUI functionality."""
    print("Testing WebUI functionality...")
    
    # Create WebUI config
    config = WebUIConfig(
        enabled=True,
        port=8765,
        host="localhost"
    )
    
    # Create managers
    webui_manager = WebUIManager(config)
    http_server = WebUIHTTPServer("localhost", 8766)
    
    print(f"Starting WebSocket server on ws://localhost:{config.port}")
    print(f"Starting HTTP server on http://localhost:8766")
    
    # Start servers
    if webui_manager.start():
        print("✓ WebSocket server started")
    else:
        print("✗ WebSocket server failed to start")
        return False
        
    if http_server.start():
        print("✓ HTTP server started")
        print("\n📡 WebUI is running!")
        print(f"🌐 Open browser to: http://localhost:8766")
        print(f"🔌 WebSocket endpoint: ws://localhost:{config.port}")
    else:
        print("✗ HTTP server failed to start")
        return False
    
    try:
        print("\n⏱️  Servers running for 10 seconds...")
        print("   (Open the browser URL above to see the WebUI)")
        print("   (Press Ctrl+C to stop early)")
        
        for i in range(10):
            time.sleep(1)
            print(f"   {10-i} seconds remaining...")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    # Cleanup
    print("\n🧹 Stopping servers...")
    webui_manager.stop()
    http_server.stop()
    print("✓ Cleanup complete")
    
    return True


if __name__ == "__main__":
    print("🤖 Keisei WebUI Test")
    print("=" * 50)
    
    success = test_webui()
    
    if success:
        print("\n✅ WebUI test completed successfully!")
        print("\nTo enable WebUI in training:")
        print("  python train.py webui.enabled=true")
    else:
        print("\n❌ WebUI test failed!")
        sys.exit(1)