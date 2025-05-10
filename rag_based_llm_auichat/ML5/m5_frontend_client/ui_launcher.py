from zenml import step
import subprocess
import os
import time
import signal
import atexit
from pathlib import Path

# Define paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.parent  # src/
# Update UI_DIR to point to the correct location where your UI files are
UI_DIR = SCRIPT_DIR / "src" / "UI"  # Updated path
REACT_UI_DIR = UI_DIR / "auichat"
PROJECT_ROOT = SCRIPT_DIR.parent  # rag_based_llm_auichat/

# Store process IDs in files for management
PID_DIR = PROJECT_ROOT / ".pids"
API_PID_FILE = PID_DIR / "flask_api.pid"
PROXY_PID_FILE = PID_DIR / "cors_proxy.pid"
REACT_PID_FILE = PID_DIR / "react_ui.pid"

def write_pid_file(pid_file, pid):
    """Write process ID to file."""
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)
    with open(pid_file, 'w') as f:
        f.write(str(pid))

def read_pid_file(pid_file):
    """Read process ID from file."""
    try:
        with open(pid_file, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

def is_process_running(pid):
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False

@step
def launch_ui_components():
    """
    Launches the Flask API, CORS Proxy, and React UI development server 
    as detached background processes so they continue running after the pipeline completes.
    """
    print("🚀 Launching UI components...")
    
    # First, verify directories exist
    if not UI_DIR.exists():
        print(f"❌ UI directory not found at: {UI_DIR}")
        print("Searching for UI directory...")
        # Try to find the UI directory
        possible_ui_dirs = list(PROJECT_ROOT.glob("**/UI"))
        if (possible_ui_dirs):
            UI_DIR_FOUND = possible_ui_dirs[0]
            REACT_UI_DIR_FOUND = UI_DIR_FOUND / "auichat"
            print(f"✅ Found UI directory at: {UI_DIR_FOUND}")
        else:
            print("❌ Could not locate UI directory. Please specify the correct path.")
            return "ui_launch_failed"
    else:
        UI_DIR_FOUND = UI_DIR
        REACT_UI_DIR_FOUND = REACT_UI_DIR
    
    # Create scripts directory to store launcher scripts
    scripts_dir = PROJECT_ROOT / ".scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Check if processes are already running
    for pid_file, service_name in [(API_PID_FILE, "Flask API"), 
                                  (PROXY_PID_FILE, "CORS Proxy"), 
                                  (REACT_PID_FILE, "React UI")]:
        pid = read_pid_file(pid_file)
        if pid and is_process_running(pid):
            print(f"✅ {service_name} is already running (PID: {pid}).")
        
    try:
        # 1. Create and run Flask API Server script
        api_script_path = scripts_dir / "launch_flask_api.sh"
        with open(api_script_path, 'w') as f:
            f.write(f"""#!/bin/bash
cd {UI_DIR_FOUND}
# Create directory for PID files if it doesn't exist
mkdir -p {PID_DIR}
nohup python api.py > {PROJECT_ROOT}/flask_api.log 2>&1 &
echo $! > {API_PID_FILE}
""")
        os.chmod(api_script_path, 0o755)
        
        print(f"🔄 Starting Flask API server from: {UI_DIR_FOUND}")
        subprocess.run(["bash", str(api_script_path)], check=True)
        api_pid = read_pid_file(API_PID_FILE)
        print(f"✅ Flask API server started (PID: {api_pid}). Log file: {PROJECT_ROOT}/flask_api.log")
        time.sleep(5)  # Give it time to initialize
        
        # 2. Create and run CORS Proxy script
        proxy_script_path = scripts_dir / "launch_cors_proxy.sh"
        with open(proxy_script_path, 'w') as f:
            f.write(f"""#!/bin/bash
cd {UI_DIR_FOUND}
# Create directory for PID files if it doesn't exist
mkdir -p {PID_DIR}
nohup python cors_proxy.py > {PROJECT_ROOT}/cors_proxy.log 2>&1 &
echo $! > {PROXY_PID_FILE}
""")
        os.chmod(proxy_script_path, 0o755)
        
        print(f"🔄 Starting CORS Proxy server from: {UI_DIR_FOUND}")
        subprocess.run(["bash", str(proxy_script_path)], check=True)
        proxy_pid = read_pid_file(PROXY_PID_FILE)
        print(f"✅ CORS Proxy server started (PID: {proxy_pid}). Log file: {PROJECT_ROOT}/cors_proxy.log")
        time.sleep(2)

        # 3. Create and run React UI script
        react_script_path = scripts_dir / "launch_react_ui.sh"
        with open(react_script_path, 'w') as f:
            f.write(f"""#!/bin/bash
cd {REACT_UI_DIR_FOUND}
# Create directory for PID files if it doesn't exist
mkdir -p {PID_DIR}
# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi
nohup npm run dev > {PROJECT_ROOT}/react_ui.log 2>&1 &
echo $! > {REACT_PID_FILE}
""")
        os.chmod(react_script_path, 0o755)
        
        print(f"🔄 Starting React UI dev server from: {REACT_UI_DIR_FOUND}")
        subprocess.run(["bash", str(react_script_path)], check=True)
        react_pid = read_pid_file(REACT_PID_FILE)
        print(f"✅ React UI dev server started (PID: {react_pid}). Log file: {PROJECT_ROOT}/react_ui.log")

        # Create a management script for the user
        mgmt_script_path = PROJECT_ROOT / "manage_ui.sh"
        with open(mgmt_script_path, 'w') as f:
            f.write(f"""#!/bin/bash
# UI Management Script

case "$1" in
  status)
    echo "Checking UI services status..."
    
    if [ -f "{API_PID_FILE}" ] && ps -p $(cat "{API_PID_FILE}") > /dev/null; then
      echo "✅ Flask API is running (PID: $(cat {API_PID_FILE}))"
    else
      echo "❌ Flask API is not running"
    fi
    
    if [ -f "{PROXY_PID_FILE}" ] && ps -p $(cat "{PROXY_PID_FILE}") > /dev/null; then
      echo "✅ CORS Proxy is running (PID: $(cat {PROXY_PID_FILE}))"
    else
      echo "❌ CORS Proxy is not running"
    fi
    
    if [ -f "{REACT_PID_FILE}" ] && ps -p $(cat "{REACT_PID_FILE}") > /dev/null; then
      echo "✅ React UI is running (PID: $(cat {REACT_PID_FILE}))"
    else
      echo "❌ React UI is not running"
    fi
    ;;
    
  stop)
    echo "Stopping UI services..."
    
    if [ -f "{API_PID_FILE}" ]; then
      kill $(cat "{API_PID_FILE}") 2>/dev/null || echo "Flask API was not running"
      rm "{API_PID_FILE}"
    fi
    
    if [ -f "{PROXY_PID_FILE}" ]; then
      kill $(cat "{PROXY_PID_FILE}") 2>/dev/null || echo "CORS Proxy was not running"
      rm "{PROXY_PID_FILE}"
    fi
    
    if [ -f "{REACT_PID_FILE}" ]; then
      kill $(cat "{REACT_PID_FILE}") 2>/dev/null || echo "React UI was not running"
      rm "{REACT_PID_FILE}"
    fi
    
    echo "All UI services stopped."
    ;;
    
  start)
    echo "Starting UI services..."
    bash "{api_script_path}"
    sleep 2
    bash "{proxy_script_path}"
    sleep 2
    bash "{react_script_path}"
    echo "All UI services started."
    ;;
    
  restart)
    $0 stop
    sleep 2
    $0 start
    ;;
    
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
esac
""")
        os.chmod(mgmt_script_path, 0o755)

        print("\n🎉 UI Components launched successfully!")
        print(f"✅ Access the UI at http://localhost:5173 (or the port specified in {PROJECT_ROOT}/react_ui.log)")
        print(f"✅ UI components will continue running after the ZenML pipeline completes.")
        print(f"✅ Use '{mgmt_script_path} status' to check status")
        print(f"✅ Use '{mgmt_script_path} stop' to stop all UI components")
        print(f"✅ Use '{mgmt_script_path} restart' to restart UI components")
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during subprocess execution: {e}")
        return "ui_launch_failed"
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return "ui_launch_failed"
            
    return "ui_launched_persistently"
