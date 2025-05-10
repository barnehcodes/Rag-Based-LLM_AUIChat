#!/usr/bin/env python3
"""
AUIChat Services Management Script

This script helps manage the various services that make up the AUIChat system.
It can be used to:
- Check the status of services
- Start/stop individual services
- Keep services running persistently
"""

import os
import sys
import subprocess
import signal
import time
import json
import socket
import argparse
from pathlib import Path

# Service configuration
SERVICES = {
    "mlflow-ui": {
        "name": "MLflow Dashboard",
        "port": 5001,
        "script_path": "notebooks/launch_mlflow_ui.sh",
        "pid_file": "notebooks/mlflow_ui.pid",
        "log_file": "notebooks/mlflow_ui.log",
        "default_enabled": True
    },
    "seldon": {
        "name": "Seldon Model Server",
        "port": None,  # Determined by Kubernetes
        "check_cmd": ["kubectl", "get", "seldondeployment", "auichat-smollm-deployment-local", "-n", "seldon"],
        "default_enabled": True
    },
    "ui": {
        "name": "Frontend UI",
        "port": 8501,  # Gradio UI typically runs on 8501
        "script_path": "demo/app.py",
        "default_enabled": True
    }
}

BASE_DIR = Path("/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat")

def is_port_in_use(port):
    """Check if a port is in use."""
    if port is None:
        return False
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def read_pid_file(pid_file_path):
    """Read PID from file if it exists."""
    try:
        with open(pid_file_path, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

def is_process_running(pid):
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False

def check_service_status(service_name, service_config):
    """Check the status of a service."""
    status = {
        "name": service_config["name"],
        "running": False,
        "details": ""
    }
    
    if "pid_file" in service_config:
        pid_file = BASE_DIR / service_config["pid_file"]
        pid = read_pid_file(pid_file)
        status["running"] = pid is not None and is_process_running(pid)
        status["details"] = f"PID: {pid}" if status["running"] else "Not running"
    
    elif "port" in service_config and service_config["port"]:
        port = service_config["port"]
        status["running"] = is_port_in_use(port)
        status["details"] = f"Port {port} is active" if status["running"] else f"Port {port} is not in use"
    
    elif "check_cmd" in service_config:
        try:
            result = subprocess.run(
                service_config["check_cmd"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            status["running"] = result.returncode == 0
            status["details"] = "Running on Kubernetes" if status["running"] else "Not deployed on Kubernetes"
        except Exception as e:
            status["running"] = False
            status["details"] = f"Error checking status: {str(e)}"
    
    return status

def start_service(service_name, service_config):
    """Start a service if it's not already running."""
    status = check_service_status(service_name, service_config)
    
    if status["running"]:
        print(f"✅ {service_config['name']} is already running. {status['details']}")
        return True
    
    print(f"🚀 Starting {service_config['name']}...")
    
    if "script_path" in service_config:
        script_path = BASE_DIR / service_config["script_path"]
        
        if service_name == "mlflow-ui" and os.path.exists(script_path.parent / "launch_mlflow_ui.sh"):
            # Special case for MLflow UI which has its own launcher script
            try:
                os.chdir(script_path.parent)
                subprocess.run(["bash", "launch_mlflow_ui.sh"], check=True)
                print(f"✅ Started {service_config['name']}. Check {service_config.get('log_file', 'logs')} for output.")
                return True
            except subprocess.CalledProcessError:
                print(f"❌ Failed to start {service_config['name']}")
                return False
                
        elif script_path.suffix == ".py":
            # For Python scripts, we'll run them with nohup
            log_file = service_name + ".log"
            try:
                os.chdir(BASE_DIR)
                nohup_cmd = f"nohup python {script_path} > {log_file} 2>&1 &"
                subprocess.run(nohup_cmd, shell=True, check=True)
                print(f"✅ Started {service_config['name']}. Check {log_file} for output.")
                return True
            except subprocess.CalledProcessError:
                print(f"❌ Failed to start {service_config['name']}")
                return False
    
    elif service_name == "seldon":
        print(f"❓ {service_config['name']} should be managed through the ZenML pipeline.")
        print("   Run the auichat_local_deployment_pipeline() to deploy it.")
        print("   This service is maintained by Kubernetes and will stay running.")
        return False
    
    print(f"❗ Don't know how to start {service_config['name']}")
    return False

def stop_service(service_name, service_config):
    """Stop a running service."""
    status = check_service_status(service_name, service_config)
    
    if not status["running"]:
        print(f"ℹ️ {service_config['name']} is not running.")
        return True
    
    print(f"🛑 Stopping {service_config['name']}...")
    
    if "pid_file" in service_config:
        pid_file = BASE_DIR / service_config["pid_file"]
        pid = read_pid_file(pid_file)
        
        if pid and is_process_running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                
                # Check if it's still running
                if is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
                
                print(f"✅ Stopped {service_config['name']} (PID: {pid})")
                return True
            except OSError as e:
                print(f"❌ Failed to stop {service_config['name']}: {e}")
                return False
    
    elif service_name == "seldon":
        print(f"❓ {service_config['name']} should be managed through Kubernetes.")
        print("   To stop it, you can use: kubectl delete seldondeployment auichat-smollm-deployment-local -n seldon")
        return False
    
    elif "port" in service_config and service_config["port"]:
        print(f"⚠️ {service_config['name']} is running on port {service_config['port']}, but I don't know its PID.")
        print(f"   You can try to kill it manually with: lsof -ti:{service_config['port']} | xargs kill")
        return False
    
    print(f"❗ Don't know how to stop {service_config['name']}")
    return False

def check_all_services():
    """Check the status of all services."""
    all_running = True
    print("\n📊 AUIChat Services Status:\n")
    
    for service_name, service_config in SERVICES.items():
        status = check_service_status(service_name, service_config)
        status_icon = "✅" if status["running"] else "❌"
        print(f"{status_icon} {service_config['name']}: {status['details']}")
        if not status["running"] and service_config["default_enabled"]:
            all_running = False
    
    if all_running:
        print("\n✅ All services are running!")
    else:
        print("\n⚠️ Some services are not running. Use --start-all to start them.")
    
    return all_running

def start_all_services():
    """Start all services that are configured to be enabled by default."""
    print("\n🚀 Starting all AUIChat services...\n")
    
    for service_name, service_config in SERVICES.items():
        if service_config["default_enabled"]:
            start_service(service_name, service_config)
    
    print("\n📊 Current status:")
    check_all_services()

def stop_all_services():
    """Stop all running services."""
    print("\n🛑 Stopping all AUIChat services...\n")
    
    for service_name, service_config in SERVICES.items():
        stop_service(service_name, service_config)
    
    print("\n📊 Current status:")
    check_all_services()

def keep_services_running():
    """Monitor services and restart them if they stop."""
    print("\n🔄 Keeping AUIChat services running until interrupted...\n")
    print("Press Ctrl+C to stop monitoring.\n")
    
    try:
        while True:
            for service_name, service_config in SERVICES.items():
                if service_config["default_enabled"]:
                    status = check_service_status(service_name, service_config)
                    if not status["running"]:
                        print(f"\n⚠️ {service_config['name']} is not running. Attempting to restart...")
                        start_service(service_name, service_config)
            
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped. Services will continue running.")
        print("Use --stop-all to stop all services.\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AUIChat Services Manager")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Check the status of all services")
    group.add_argument("--start-all", action="store_true", help="Start all services")
    group.add_argument("--stop-all", action="store_true", help="Stop all services")
    group.add_argument("--keep-running", action="store_true", help="Keep checking and restarting services")
    
    # Actions for individual services
    for service_name, service_config in SERVICES.items():
        group.add_argument(f"--start-{service_name}", action="store_true", help=f"Start {service_config['name']}")
        group.add_argument(f"--stop-{service_name}", action="store_true", help=f"Stop {service_config['name']}")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Change to the base directory
    os.chdir(BASE_DIR)
    
    # Handle global actions
    if args.check or len(sys.argv) == 1:  # Default action is to check
        check_all_services()
        return
    
    if args.start_all:
        start_all_services()
        return
    
    if args.stop_all:
        stop_all_services()
        return
    
    if args.keep_running:
        start_all_services()  # First make sure all services are started
        keep_services_running()
        return
    
    # Handle individual service actions
    for service_name, service_config in SERVICES.items():
        start_arg = f"start_{service_name.replace('-', '_')}"
        stop_arg = f"stop_{service_name.replace('-', '_')}"
        
        if hasattr(args, start_arg) and getattr(args, start_arg):
            start_service(service_name, service_config)
            return
        
        if hasattr(args, stop_arg) and getattr(args, stop_arg):
            stop_service(service_name, service_config)
            return

if __name__ == "__main__":
    main()