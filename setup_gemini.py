"""
Quick setup script for Gemini API integration.
Helps users set up the GEMINI_API_KEY environment variable.
"""

import os
import sys


def check_gemini_setup():
    """Check if Gemini API is properly configured."""
    api_key = os.getenv('GEMINI_API_KEY')
    
    if api_key:
        # Mask API key for display
        masked_key = api_key[:8] + '*' * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else '*' * len(api_key)
        print("✅ GEMINI_API_KEY is set!")
        print(f"   Key: {masked_key}")
        return True
    else:
        print("❌ GEMINI_API_KEY is not set!")
        return False


def test_gemini_api():
    """Test Gemini API connection."""
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package is installed")
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ Cannot test API without GEMINI_API_KEY")
            return False
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Simple test
            response = model.generate_content("Say 'API working' if you can read this.")
            print("✅ Gemini API is working!")
            print(f"   Test response: {response.text[:50]}...")
            return True
        
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return False
    
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("   Install with: pip install google-generativeai")
        return False


def main():
    print("="*80)
    print("🤖 GEMINI API SETUP CHECK")
    print("="*80)
    print()
    
    # Check if API key is set
    key_set = check_gemini_setup()
    print()
    
    # Test package installation and API
    if key_set:
        test_gemini_api()
    
    print()
    print("="*80)
    
    # Provide instructions if not set up
    if not key_set:
        print("\n📖 HOW TO SET UP GEMINI API KEY:")
        print("="*80)
        print()
        print("1. Get your API key from: https://aistudio.google.com/app/apikey")
        print()
        print("2. Set the environment variable:")
        print()
        
        if sys.platform == 'win32':
            print("   Windows PowerShell (Current Session):")
            print('   $env:GEMINI_API_KEY="your-api-key-here"')
            print()
            print("   Windows PowerShell (Permanent):")
            print("   [System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your-api-key-here', 'User')")
            print()
            print("   Windows Command Prompt:")
            print('   set GEMINI_API_KEY=your-api-key-here')
        else:
            print("   Linux/Mac (Current Session):")
            print('   export GEMINI_API_KEY="your-api-key-here"')
            print()
            print("   Linux/Mac (Permanent - add to ~/.bashrc or ~/.zshrc):")
            print('   echo \'export GEMINI_API_KEY="your-api-key-here"\' >> ~/.bashrc')
        
        print()
        print("3. Install required package:")
        print("   pip install google-generativeai")
        print()
        print("4. Run this script again to test:")
        print("   python setup_gemini.py")
        print()
        print("="*80)
    else:
        print("\n✅ SETUP COMPLETE")
        print("="*80)
        print()
        print("You can now use Gemini AI analysis with:")
        print()
        print("  python example_patchcore_sam_reports.py \\")
        print("    --image-a samples/ref.jpg \\")
        print("    --image-b samples/test.jpg \\")
        print("    --output-dir reports/ \\")
        print("    --use-gemini")
        print()
        print("Or in Python code:")
        print()
        print("  from pipelines.anomaly_patchcore_sam import run_patchcore_sam_pipeline")
        print()
        print("  result = run_patchcore_sam_pipeline(")
        print("      test_img=image_b,")
        print("      ref_img=image_a,")
        print("      generate_reports=True,")
        print("      output_dir='reports/',")
        print("      use_gemini=True")
        print("  )")
        print()
        print("="*80)


if __name__ == '__main__':
    main()
