#!/usr/bin/env python3
"""
Generate SVG piece images for Shogi WebUI
Creates clean, scalable piece images instead of Unicode text
"""

import os

def create_piece_svg(piece_name, symbol, color, promoted=False):
    """Create an SVG piece image"""
    
    # Color scheme
    if color == 'black':
        bg_color = '#f5f5dc'  # Beige background for black pieces
        text_color = '#000000'  # Black text
        border_color = '#8b4513'  # Brown border
    else:
        bg_color = '#f0e6d2'  # Slightly different beige for white pieces  
        text_color = '#8b0000'  # Dark red text
        border_color = '#8b4513'  # Brown border
        
    # Font size based on symbol length
    if len(symbol) == 1:
        font_size = '24'
        y_pos = '28'
    else:  # Multi-character promoted pieces
        font_size = '16'
        y_pos = '26'
    
    # Add promotion indicator
    promotion_marker = ''
    if promoted:
        promotion_marker = f'<circle cx="8" cy="8" r="3" fill="{text_color}" opacity="0.3"/>'
    
    svg_content = f'''<svg width="40" height="40" xmlns="http://www.w3.org/2000/svg">
  <!-- Piece background -->
  <rect x="2" y="2" width="36" height="36" rx="4" fill="{bg_color}" stroke="{border_color}" stroke-width="1"/>
  
  <!-- Promotion indicator -->
  {promotion_marker}
  
  <!-- Piece symbol -->
  <text x="20" y="{y_pos}" font-family="serif" font-size="{font_size}" font-weight="bold" 
        text-anchor="middle" fill="{text_color}">{symbol}</text>
        
  <!-- White piece rotation indicator -->
  {"<!-- Rotated -->" if color == 'white' else ""}
</svg>'''
    
    return svg_content

def generate_all_pieces():
    """Generate all piece images"""
    
    pieces = {
        'pawn': { 'black': '歩', 'white': '歩' },
        'lance': { 'black': '香', 'white': '香' },
        'knight': { 'black': '桂', 'white': '桂' },
        'silver': { 'black': '銀', 'white': '銀' },
        'gold': { 'black': '金', 'white': '金' },
        'bishop': { 'black': '角', 'white': '角' },
        'rook': { 'black': '飛', 'white': '飛' },
        'king': { 'black': '王', 'white': '王' },
        'promoted_pawn': { 'black': 'と', 'white': 'と' },
        'promoted_lance': { 'black': '成香', 'white': '成香' },
        'promoted_knight': { 'black': '成桂', 'white': '成桂' },
        'promoted_silver': { 'black': '成銀', 'white': '成銀' },
        'promoted_bishop': { 'black': '馬', 'white': '馬' },
        'promoted_rook': { 'black': '龍', 'white': '龍' }
    }
    
    # Create images directory
    images_dir = '/home/john/keisei/keisei/webui/static/images'
    os.makedirs(images_dir, exist_ok=True)
    
    generated_files = []
    
    for piece_type, colors in pieces.items():
        promoted = piece_type.startswith('promoted_')
        
        for color, symbol in colors.items():
            filename = f'{piece_type}_{color}.svg'
            filepath = os.path.join(images_dir, filename)
            
            svg_content = create_piece_svg(piece_type, symbol, color, promoted)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(svg_content)
                
            generated_files.append(filename)
            print(f"✅ Created: {filename} ({symbol})")
    
    return generated_files

if __name__ == '__main__':
    print("🎨 Generating Shogi piece images...")
    files = generate_all_pieces()
    print(f"\n✅ Generated {len(files)} piece images in keisei/webui/static/images/")
    print("\nNext steps:")
    print("1. Update app.js to use <img> tags instead of text")
    print("2. Update CSS for proper image sizing")
    print("3. Test the updated board rendering")