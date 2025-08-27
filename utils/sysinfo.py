import platform
import psutil
import subprocess
import multiprocessing
import pynvml
import os
import json
import sys
from datetime import datetime
from importlib import metadata

# TODO optimize "import" statements across the codebase
# TODO Merge this with sysinfo.py

# Default safety percentages for resource limits
# Adjust as needed for more conservative or aggressive defaults
DEFAULT_VRAM_PCT = 0.5
DEFAULT_RAM_PCT = 0.5
DEFAULT_CPU_PCT = 0.5

def get_system_info(include_limits=True,
                    vram_pct = DEFAULT_VRAM_PCT,
                    ram_pct = DEFAULT_RAM_PCT,
                    cpu_pct = DEFAULT_CPU_PCT,
                    export_hardware_info = True,
                    export_performance_info = False,
                    console_logging = True):
    """
    Consolidated system information gathering with optional resource limits
    
    Args:
        include_limits: Whether to calculate recommended resource limits
        vram_pct: Percentage of VRAM to use as limit (0.0-1.0)
        ram_pct: Percentage of RAM to use as limit (0.0-1.0)  
        cpu_pct: Percentage of CPU threads to use as limit (0.0-1.0)
        
    
    Returns:
        dict: System information and optional resource limits
    """
    full_info = {}
    
    #full_infoBasic system information
    full_info.update({
        "OS": platform.platform(),
        "CPU": platform.processor(),
        "Cores": psutil.cpu_count(logical=True),
        "Physical_Cores": psutil.cpu_count(logical=False),
        "RAM_GB": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "RAM_Available_GB": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "Python_Version": platform.python_version(),
        "Architecture": platform.architecture()[0],
    })
    
    # GPU Information - Try multiple methods
    gpu_info = _get_gpu_info()
    full_info.update(gpu_info)
    
    # Python Environment Information
    env_info = _detect_environment_info()
    full_info["Python_Environment"] = env_info
    
    # Resource limits (if requested)
    if include_limits:
        limits = _calculate_resource_limits(vram_pct, ram_pct, cpu_pct)
        full_info["Resource_Limits"] = limits
    
    return full_info

def _get_gpu_info():
    """Get GPU information using multiple fallback methods"""
    gpu_info = {}
    
    # Method 1: Try NVIDIA Management Library (most reliable for NVIDIA GPUs)
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        gpu_list = []
        total_vram = 0
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = round(memory_info.total / (1024 ** 3), 2)
            
            gpu_list.append(f"{name} ({vram_gb}GB)")
            total_vram += memory_info.total
        
        gpu_info["GPU_Count"] = device_count
        gpu_info["GPUs"] = gpu_list
        gpu_info["Total_VRAM_GB"] = round(total_vram / (1024 ** 3), 2)
        gpu_info["GPU_Method"] = "NVIDIA-ML"
        
        pynvml.nvmlShutdown()
        return gpu_info
        
    except Exception as e:
        gpu_info["NVIDIA_ML_Error"] = str(e)
    
    # Method 2: Try DirectX Diagnostic (Windows only)
    if platform.system() == "Windows":
        try:
            # Use a temporary file in system temp directory
            temp_file = os.path.join(os.environ.get('TEMP', '.'), 'meshviz_dxdiag.txt')
            
            result = subprocess.run(
                ["dxdiag", "/t", temp_file], 
                check=True, 
                timeout=30,
                capture_output=True
            )
            
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8", errors="ignore") as f:
                    dxdiag_data = f.read()
                
                # Extract GPU name from dxdiag output
                lines = dxdiag_data.split('\n')
                for line in lines:
                    if "Card name:" in line:
                        gpu_name = line.split("Card name:")[1].strip()
                        gpu_info["GPU"] = gpu_name
                        break
                
                # Clean up temp file
                os.remove(temp_file)
                gpu_info["GPU_Method"] = "DirectX-Diag"
                return gpu_info
                
        except Exception as e:
            gpu_info["DirectX_Diag_Error"] = str(e)
    
    # Method 3: Fallback - basic detection
    gpu_info["GPU"] = "Unable to detect GPU automatically"
    gpu_info["GPU_Method"] = "None"
    gpu_info["GPU_Note"] = "Install nvidia-ml-py for NVIDIA GPU detection"
    
    return gpu_info

def _calculate_resource_limits(vram_pct = DEFAULT_VRAM_PCT,
                               ram_pct = DEFAULT_RAM_PCT,
                               cpu_pct = DEFAULT_CPU_PCT):
    """Calculate recommended resource usage limits"""
    limits = {}
    
    # CPU limits
    total_threads = multiprocessing.cpu_count()
    limits["max_threads"] = int(total_threads * cpu_pct)
    limits["total_threads"] = total_threads
    
    # RAM limits  
    total_ram = psutil.virtual_memory().total
    limits["max_ram_bytes"] = int(total_ram * ram_pct)
    limits["max_ram_gb"] = round(limits["max_ram_bytes"] / (1024 ** 3), 2)
    limits["total_ram_gb"] = round(total_ram / (1024 ** 3), 2)
    
    # GPU limits (NVIDIA only for now)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Primary GPU
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        limits["max_vram_bytes"] = int(memory_info.total * vram_pct)
        limits["max_vram_gb"] = round(limits["max_vram_bytes"] / (1024 ** 3), 2)
        limits["total_vram_gb"] = round(memory_info.total / (1024 ** 3), 2)
        
        pynvml.nvmlShutdown()
        
    except Exception as e:
        limits["vram_detection_error"] = str(e)
        limits["max_vram_gb"] = "Unable to detect"
    
    # Add percentage settings used
    limits["percentages"] = {
        "cpu": cpu_pct,
        "ram": ram_pct, 
        "vram": vram_pct
    }
    
    return limits

def get_system_limits(vram_pct = DEFAULT_VRAM_PCT, 
                      ram_pct = DEFAULT_RAM_PCT, 
                      cpu_pct = DEFAULT_CPU_PCT):
    """"
    Legacy support function wrapper to extract just the resource limits from full system info
    Args:   
        vram_pct: Percentage of VRAM to use as limit (0.0-1.0)
        ram_pct: Percentage of RAM to use as limit (0.0-1.0)  
        cpu_pct: Percentage of CPU threads to use as limit (0.0-1.0)
    Returns:
        dict: Resource limits with keys "threads", "ram", "vram"
    """

    # Call the enhanced function internally with those percentages
    full_info = get_system_info(include_limits=True, 
                               vram_pct=vram_pct, 
                               ram_pct=ram_pct, 
                               cpu_pct=cpu_pct)
    
    # Step 2: Extract the old-format data using your nested access pattern
    return {
        "threads": full_info['Resource_Limits']["max_threads"],
        "ram": full_info['Resource_Limits']["max_ram_bytes"], 
        "vram": full_info['Resource_Limits'].get("max_vram_bytes", None)
    }

def export_system_info(full_info,
                       filename = "system_info.json", 
                       filepath = None):
    """
    Export system information to a json file
    Args:
        full_info: dict of system information
        filename: output file name
        filepath: optional custom path (default: project_root/contexts)

        Returns:
            booolean: success status
    """
    exit_code = True # Assume success unless error occurs
    
    # If no filepath provided, use project root + contexts folder
    if filepath is None:
        # Get the directory where sysinfo.py lives, then go up one level to project root
        script_dir = os.path.dirname(__file__)  # D:\Repos\Meshviz\utils
        project_root = os.path.dirname(script_dir)  # D:\Repos\Meshviz  
        filepath = os.path.join(project_root, "contexts")  # D:\Repos\Meshviz\contexts
    
    # Ensure directory exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    full_path = os.path.join(filepath, filename)
    print(f"System information exported to {full_path}")
    
    # Write to file
    try:
        with open(full_path, "w") as f:
            # Convert dict to formatted JSON string
            json_data = json.dumps(full_info, indent=4)
            f.write(json_data)
            
    except Exception as e:
        print(f"Error exporting system info: {e}")
        exit_code = False

    return exit_code

BASIC_PACKAGES = [
    "numpy", "scipy", "pandas", "matplotlib", "seaborn", "plotly",
    "sympy", "numba", "psutil", "pynvml", "glfw"]

def _detect_key_packages(pckg_list=BASIC_PACKAGES, all_pckgs=False):
    """Gather the information of available packages"""
    
    if all_pckgs:
        # Get all installed packages with versions
        all_packages = {}
        for dist in metadata.distributions():
            all_packages[dist.metadata['Name']] = dist.version
        return all_packages
    
    # Get specific packages only
    pckg_versions = {}
    for pckg in pckg_list:
        try:
            pckg_versions[pckg] = metadata.version(pckg)
        except metadata.PackageNotFoundError:
            pckg_versions[pckg] = "Not installed"
    
    return pckg_versions

def _detect_environment_info():
    """Get Python environment information"""
    return {
        "python_executable": sys.executable,
        "key_packages": _detect_key_packages(),
        "project_root": os.path.dirname(os.path.dirname(__file__))
    }

# Example usage:
if __name__ == "__main__":
    print("=== Testing System Info Detection ===")
    
    # Get full system info with limits
    full_info = get_system_info()
    print(f"System: {full_info['OS']}")
    print(f"Python: {full_info['Python_Version']} at {full_info['Python_Environment']['python_executable']}")
    print(f"GPU: {full_info.get('GPUs', 'Unknown')}")
    
    print("\n=== Testing Package Detection ===")
    packages = _detect_key_packages()
    print("Key packages:")
    for pkg, version in packages.items():
        print(f"  {pkg}: {version}")
    
    print("\n=== Testing JSON Export ===")
    success = export_system_info(full_info, "test_complete_system_info.json")
    print(f"Export successful: {success}")