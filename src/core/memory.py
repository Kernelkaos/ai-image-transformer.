import gc
import logging
import torch

logger = logging.getLogger(__name__)

def flush_vram():
    """Forcefully releases unused VRAM cache and runs CPU garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logger.debug("VRAM and GC cache cleared.")

def get_vram_info():
    """Retrieves current VRAM usage details in Megabytes."""
    if not torch.cuda.is_available():
        return {"device": "CPU", "allocated": 0, "cached": 0, "total": 0}
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    cached = torch.cuda.memory_reserved(device) / (1024 ** 2)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    
    return {
        "device": torch.cuda.get_device_name(device),
        "allocated_mb": round(allocated, 2),
        "cached_mb": round(cached, 2),
        "total_mb": round(total, 2)
    }

class VRAMGuard:
    """Context manager to ensure VRAM is cleared before and after executing a block."""
    def __enter__(self):
        flush_vram()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        flush_vram()
