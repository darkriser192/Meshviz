import platform
import psutil
import subprocess
import multiprocessing
import pynvml

def get_system_limits(vram_pct=0.85, ram_pct=0.80, cpu_pct=0.90):
    # CPU
    total_threads = multiprocessing.cpu_count()
    max_threads = int(total_threads * cpu_pct)

    # RAM
    total_ram = psutil.virtual_memory().total
    max_ram = int(total_ram * ram_pct)

    # GPU
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # assume first GPU
    total_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    max_vram = int(total_vram * vram_pct)

    return {
        "threads": max_threads,
        "ram": max_ram,
        "vram": max_vram
    }

def get_system_info():
    info = {
        "OS": platform.platform(),
        "CPU": platform.processor(),
        "Cores": psutil.cpu_count(logical=True),
        "RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }

    # Try to get GPU info via DirectX diag (Windows specific)
    try:
        result = subprocess.run(
            ["dxdiag", "/t", "dxdiag_output.txt"], check=True
        )
        with open("dxdiag_output.txt", "r", encoding="utf-8", errors="ignore") as f:
            dxdiag_data = f.read()
        info["GPU"] = "Check dxdiag_output.txt for details."
    except Exception:
        info["GPU"] = "Unable to fetch GPU info automatically."

    return info
