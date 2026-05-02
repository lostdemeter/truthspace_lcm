#!/usr/bin/env python3
"""
Test the φ-Chat with the Valentine's Day card example.
This version includes error recovery - if the first attempt fails,
it feeds the error back to the model to fix.
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm/phi_chat')

from chat import PhiChat, OUTPUT_DIR

def test_valentine_with_recovery():
    """Test creating a Valentine's Day card with error recovery."""
    print("=" * 60)
    print("φ-Chat Valentine's Day Card Test (with Error Recovery)")
    print("=" * 60)
    
    chat = PhiChat()
    
    # The user's request
    request = """Create a Valentine's Day card to 'Katie my Love' using matplotlib and save the output as valentine_card.png

IMPORTANT: 
- Draw the heart shape using math (parametric equations), don't load external images
- Make it beautiful with nice colors and styling
- The code must be completely self-contained"""
    
    print(f"\nUser: {request}\n")
    
    response, execution = chat.chat(request)
    
    # If it failed, try to recover
    max_retries = 2
    retry_count = 0
    
    while execution and not execution.success and retry_count < max_retries:
        retry_count += 1
        print(f"\n🔄 Attempt {retry_count + 1}: Asking model to fix the error...")
        
        error_msg = execution.stderr
        fix_request = f"""The code failed with this error:

{error_msg}

Please fix the code. Remember:
- Don't try to load external files
- Draw shapes using matplotlib primitives or math equations
- Make sure all imports are included"""
        
        response, execution = chat.chat(fix_request)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    if execution:
        if execution.success:
            print("✓ Code executed successfully!")
            if execution.files_created:
                print(f"✓ Files created: {execution.files_created}")
                for f in execution.files_created:
                    if 'valentine' in f.lower() or 'png' in f.lower():
                        print(f"\n🎉 Valentine's Day card created: {f}")
            else:
                # Check if file exists in output dir
                import os
                png_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
                if png_files:
                    print(f"✓ PNG files in output: {png_files}")
            print(f"\nRetries needed: {retry_count}")
        else:
            print("✗ Execution failed after all retries")
            print(f"Error: {execution.stderr}")
    else:
        print("✗ No code was generated")
    
    return execution and execution.success


if __name__ == "__main__":
    success = test_valentine_with_recovery()
    sys.exit(0 if success else 1)
