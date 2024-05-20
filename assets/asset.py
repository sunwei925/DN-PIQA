import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
import models.FaceIQA as FaceIQA
import models.LIQE as LIQE
from thop import profile # pip install thop

def torch_cuda_memory_usage():
    """Returns CUDA memory usage if available"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
        allocated_memory = torch.cuda.memory_allocated()  # Total allocated memory
        cached_memory = torch.cuda.memory_reserved()  # Total cached memory
        return allocated_memory / (1024**3), cached_memory / (1024**3)  # Convert bytes to GB
    else:
        return 0, 0

def test_model_resources(model, batch):
    
    macs, params = profile(model, inputs=(batch, ), verbose=False)
    flops = macs * 2  # Convert MACs to FLOPs
    tflops = flops / (10**12)  # Convert FLOPs to TFLOPs  
    
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats for accurate peak measurement

    # Measure memory before inference
    allocated_before, cached_before = torch_cuda_memory_usage()
    
    model = model.cuda()  # Move model to GPU
    batch = batch.cuda()  # Move data to GPU
    
    # Dummy forward pass to measure VRAM usage
    with torch.no_grad():
        _ = model(batch)
        
    # Measure memory after inference
    allocated_after, cached_after = torch_cuda_memory_usage()
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # Peak allocated memory during inference
    
    vram_usage_allocated = allocated_after - allocated_before  # Approximation of additional VRAM used during inference
    vram_usage_cached = cached_after - cached_before  # Approximation based on cached memory

    print(f"Paramaters: {params}")
    print(f"MACs: {macs}")
    print(f"FLOPs: {flops}")
    print(f"TFLOPs: {tflops}")
    print(f"Approx. Additional VRAM Usage (Allocated) during Inference: {vram_usage_allocated} GB")
    print(f"Approx. Additional VRAM Usage (Cached) during Inference: {vram_usage_cached} GB")
    print(f"Peak VRAM Usage during Inference: {peak_allocated} GB")
    
    del model, batch  # Free up memory
    torch.cuda.empty_cache()  # Clear cache

def test_model_resources_FaceIQA(model, batch, batch_face, batch_feature):
    
    macs, params = profile(model, inputs=(batch, batch_face, batch_feature, ), verbose=False)
    flops = macs * 2  # Convert MACs to FLOPs
    tflops = flops / (10**12)  # Convert FLOPs to TFLOPs  
    
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats for accurate peak measurement

    # Measure memory before inference
    allocated_before, cached_before = torch_cuda_memory_usage()
    
    model = model.cuda()  # Move model to GPU
    batch = batch.cuda()  # Move data to GPU
    batch_face = batch_face.cuda()  # Move data to GPU
    batch_feature = batch_feature.cuda()  # Move data to GPU

    
    # Dummy forward pass to measure VRAM usage
    with torch.no_grad():
        _ = model(batch, batch_face, batch_feature)
        
    # Measure memory after inference
    allocated_after, cached_after = torch_cuda_memory_usage()
    peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # Peak allocated memory during inference
    
    vram_usage_allocated = allocated_after - allocated_before  # Approximation of additional VRAM used during inference
    vram_usage_cached = cached_after - cached_before  # Approximation based on cached memory

    print(f"Paramaters: {params}")
    print(f"MACs: {macs}")
    print(f"FLOPs: {flops}")
    print(f"TFLOPs: {tflops}")
    print(f"Approx. Additional VRAM Usage (Allocated) during Inference: {vram_usage_allocated} GB")
    print(f"Approx. Additional VRAM Usage (Cached) during Inference: {vram_usage_cached} GB")
    print(f"Peak VRAM Usage during Inference: {peak_allocated} GB")
    
    del model, batch  # Free up memory
    torch.cuda.empty_cache()  # Clear cache

# Tes the computation complexity of LIQE
model = LIQE.LIQE_feature()
batch_size = 1 # Test the batch size you want
batch = torch.stack([torch.randn(3, 1280, 960)]*batch_size)

test_model_resources(model, batch)

# Tes the computation complexity of FaceIQA
model_FaceIQA = FaceIQA.PIQ_model(pretrained_path = None, pretrained_path_face= None)
batch_FaceIQA_size = 1 # Test the batch size you want
batch_FaceIQA = torch.stack([torch.randn(3, 384, 384)]*batch_size)
batch_FaceIQA_face = torch.stack([torch.randn(3, 384, 384)]*batch_size)
batch_FaceIQA_feature = torch.stack([torch.randn(495)]*batch_size)

test_model_resources_FaceIQA(model_FaceIQA, batch_FaceIQA, batch_FaceIQA_face, batch_FaceIQA_feature)