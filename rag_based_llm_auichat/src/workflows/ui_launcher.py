\
from zenml import step
import subprocess
import os
import time
from pathlib import Path

# Define paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.parent # src/
UI_DIR = SCRIPT_DIR / "UI"
REACT_UI_DIR = UI_DIR / "auichat"
PROJECT_ROOT = SCRIPT_DIR.parent # rag_based_llm_auichat/

@step
def launch_ui_components():
    
    # Launches the Flask API, CORS Proxy, and React UI development server 
    # as background processes.

    print("🚀 Launching UI components...")
    
    api_process = None
    proxy_process = None
    react_process = None
    
    try:
        # 1. Start Flask API Server
        print(f"🔄 Starting Flask API server from: {UI_DIR}")
        api_command = ["python", "api.py"]
        api_process = subprocess.Popen(api_command, cwd=str(UI_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"✅ Flask API server started (PID: {api_process.pid}). Waiting for initialization...")
        # Give it some time to initialize (adjust if needed)
        time.sleep(15) 
        
        # 2. Start CORS Proxy
        print(f"🔄 Starting CORS Proxy server from: {UI_DIR}")
        proxy_command = ["python", "cors_proxy.py"]
        proxy_process = subprocess.Popen(proxy_command, cwd=str(UI_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"✅ CORS Proxy server started (PID: {proxy_process.pid}).")
        time.sleep(2)

        # 3. Start React UI (Vite dev server)
        print(f"🔄 Starting React UI dev server from: {REACT_UI_DIR}")
        # Check if node_modules exists, run npm install if not
        if not (REACT_UI_DIR / "node_modules").exists():
            print("node_modules not found. Running 'npm install'...")
            install_process = subprocess.run(["npm", "install"], cwd=str(REACT_UI_DIR), check=True, capture_output=True, text=True)
            print("npm install completed.")
            print(install_process.stdout)
        
        react_command = ["npm", "run", "dev"]
        react_process = subprocess.Popen(react_command, cwd=str(REACT_UI_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"✅ React UI dev server started (PID: {react_process.pid}).")
        print("\\n🎉 UI Components launched. Access the UI at http://localhost:5173 (or the port specified by Vite).")
        print("   Keep this ZenML step running to keep the UI active.")
        print("   API Output:")
        # Non-blocking read for API output (optional)
        # for line in iter(api_process.stdout.readline, ''):
        #     print(f"[API]: {line.strip()}")
        
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
