#!/usr/bin/env python3
"""
Test the φ-Chat with the Valentine's Day card example.
Non-interactive test to verify it works.
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat')

from chat import PhiChat, OUTPUT_DIR

def test_valentine():
    """Test creating a Valentine's Day card."""
    print("=" * 60)
    print("φ-Chat Valentine's Day Card Test")
    print("=" * 60)
    
    chat = PhiChat()
    
    # The user's request
    request = "Create a Valentine's Day card to 'Katie my Love' using matplotlib and save the output as valentine_card.png"
    
    print(f"\nUser: {request}\n")
    
    response, execution = chat.chat(request)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    if execution:
        if execution.success:
            print("✓ Code executed successfully!")
            if execution.files_created:
                print(f"✓ Files created: {execution.files_created}")
                # Check if the valentine card was created
                for f in execution.files_created:
                    if 'valentine' in f.lower() or 'png' in f.lower():
                        print(f"\n🎉 Valentine's Day card created: {f}")
            else:
                # Check if file exists in output dir
                import os
                png_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
                if png_files:
                    print(f"✓ PNG files in output: {png_files}")
        else:
            print("✗ Execution failed")
            print(f"Error: {execution.stderr}")
    else:
        print("✗ No code was generated")
    
    return execution and execution.success


if __name__ == "__main__":
    success = test_valentine()
    sys.exit(0 if success else 1)
