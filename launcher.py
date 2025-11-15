"""
Simple launcher script for F1 Visual Difference Engine
Provides easy commands to run the system
"""
import os
import sys
import subprocess

def print_menu():
    print("\n" + "="*60)
    print("🏎️  F1 VISUAL DIFFERENCE ENGINE - LAUNCHER")
    print("="*60)
    print("\n1. Run Streamlit Dashboard (Recommended)")
    print("2. Run Installation Test")
    print("3. Run Command Line Demo (back1 vs back2)")
    print("4. Open Jupyter Notebook")
    print("5. Exit")
    print("\n" + "="*60)

def run_streamlit():
    print("\n🚀 Launching Streamlit Dashboard...")
    print("This will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "demo/streamlit_app.py"])

def run_test():
    print("\n🧪 Running Installation Test...")
    subprocess.run([sys.executable, "test_installation.py"])

def run_demo():
    print("\n🏎️ Running Command Line Demo...")
    subprocess.run([sys.executable, "main_pipeline.py", "samples/back1.jpeg", "samples/back2.jpeg"])

def run_jupyter():
    print("\n📓 Opening Jupyter Notebook...")
    subprocess.run([sys.executable, "-m", "jupyter", "notebook", "demo/app.ipynb"])

def main():
    while True:
        print_menu()
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            run_streamlit()
        elif choice == "2":
            run_test()
        elif choice == "3":
            run_demo()
        elif choice == "4":
            run_jupyter()
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please select 1-5.")
        
        if choice in ["2", "3"]:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
