import platform
import psutil
import subprocess
import multiprocessing
import pynvml
import os

# TODO optimize "import" statements across the codebase
# TODO Merge this with sysinfo.py

def get_system_info(include_limits=True, vram_pct=0.85, ram_pct=0.80, cpu_pct=0.90):
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
    info = {}
    
    # Basic system information
    info.update({
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
    info.update(gpu_info)
    
    # Resource limits (if requested)
    if include_limits:
        limits = _calculate_resource_limits(vram_pct, ram_pct, cpu_pct)
        info["Resource_Limits"] = limits
    
    return info

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

def _calculate_resource_limits(vram_pct, ram_pct, cpu_pct):
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

# Example usage:
if __name__ == "__main__":
    # Get full system info with limits
    full_info = get_system_info()
    
    # Get basic info without limits
    basic_info = get_system_info(include_limits=False)
    
    # Custom limit percentages
    conservative_info = get_system_info(vram_pct=0.70, ram_pct=0.60, cpu_pct=0.75)