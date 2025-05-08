from zenml import step
import subprocess
import os
import time
from pathlib import Path

# Define paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.parent  # src/
# Update UI_DIR to point to the correct location where your UI files are
UI_DIR = SCRIPT_DIR / "src" / "UI"  # Updated path
REACT_UI_DIR = UI_DIR / "auichat"
PROJECT_ROOT = SCRIPT_DIR.parent  # rag_based_llm_auichat/

@step
def launch_ui_components():
    """
    Launches the Flask API, CORS Proxy, and React UI development server 
    as background processes.
    """
    print("🚀 Launching UI components...")
    
    # First, verify directories exist
    if not UI_DIR.exists():
        print(f"❌ UI directory not found at: {UI_DIR}")
        print("Searching for UI directory...")
        # Try to find the UI directory
        possible_ui_dirs = list(PROJECT_ROOT.glob("**/UI"))
        if possible_ui_dirs:
            UI_DIR_FOUND = possible_ui_dirs[0]
            REACT_UI_DIR_FOUND = UI_DIR_FOUND / "auichat"
            print(f"✅ Found UI directory at: {UI_DIR_FOUND}")
        else:
            print("❌ Could not locate UI directory. Please specify the correct path.")
            return "ui_launch_failed"
    else:
        UI_DIR_FOUND = UI_DIR
        REACT_UI_DIR_FOUND = REACT_UI_DIR
    
    api_process = None
    proxy_process = None
    react_process = None
    
    try:
        # 1. Start Flask API Server
        print(f"🔄 Starting Flask API server from: {UI_DIR_FOUND}")
        api_command = ["python", "api.py"]
        api_process = subprocess.Popen(api_command, cwd=str(UI_DIR_FOUND), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"✅ Flask API server started (PID: {api_process.pid}). Waiting for initialization...")
        # Give it some time to initialize (adjust if needed)
        time.sleep(10) 
        
        # 2. Start CORS Proxy
        print(f"🔄 Starting CORS Proxy server from: {UI_DIR_FOUND}")
        proxy_command = ["python", "cors_proxy.py"]
        proxy_process = subprocess.Popen(proxy_command, cwd=str(UI_DIR_FOUND), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"✅ CORS Proxy server started (PID: {proxy_process.pid}).")
        time.sleep(2)

        # 3. Start React UI (Vite dev server)
        print(f"🔄 Starting React UI dev server from: {REACT_UI_DIR_FOUND}")
        # Check if node_modules exists, run npm install if not
        if not (REACT_UI_DIR_FOUND / "node_modules").exists():
            print("node_modules not found. Running 'npm install'...")
            install_process = subprocess.run(["npm", "install"], cwd=str(REACT_UI_DIR_FOUND), check=True, capture_output=True, text=True)
            print("npm install completed.")
            print(install_process.stdout)
        
        react_command = ["npm", "run", "dev"]
        react_process = subprocess.Popen(react_command, cwd=str(REACT_UI_DIR_FOUND), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"✅ React UI dev server started (PID: {react_process.pid}).")
        print("\n🎉 UI Components launched. Access the UI at http://localhost:5173 (or the port specified by Vite).")
        print("   Keep this ZenML step running to keep the UI active.")
        
        # Keep the step running - Note: This might block pipeline completion in some runners.
        # A more robust solution might involve detaching these processes or using a dedicated service manager.
        while True: 
            time.sleep(60)
            if api_process.poll() is not None:
                print("🚨 Flask API process terminated.")
                break
            if proxy_process.poll() is not None:
                print("🚨 CORS Proxy process terminated.")
                break
            if react_process.poll() is not None:
                print("🚨 React UI process terminated.")
                break
                
    except FileNotFoundError as e:
        print(f"❌ Error: Command not found. Make sure Python and Node.js/npm are installed and in PATH. Details: {e}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during subprocess execution: {e}")
        print(f"Stderr: {e.stderr}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        print("🛑 Shutting down UI components...")
        if react_process and react_process.poll() is None:
            react_process.terminate()
            react_process.wait()
            print("React UI stopped.")
        if proxy_process and proxy_process.poll() is None:
            proxy_process.terminate()
            proxy_process.wait()
            print("CORS Proxy stopped.")
        if api_process and api_process.poll() is None:
            api_process.terminate()
            api_process.wait()
            print("Flask API stopped.")
            
    return "ui_launched" # Or potentially return PIDs/status
