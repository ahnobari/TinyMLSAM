import torch
import bitsandbytes as bnb
from segment_anything import sam_model_registry, SamPredictor
import random
import gc
import copy


from ..InferenceUtils import *

def model_quantization(orig_model, quant_type="nf4", pec=100):
    '''
    bnb quantization
    orig_model:  Either GDino or SAM2 for now
    quant_type:  "nf4" (default) or "fp4" 
    pec:         the # of percentage of total layers for quantization, 100 for quantizing the whole model
    
    '''

    if isinstance(orig_model, SAM2):
        # Original model size visualization
        original_size = calculate_model_size(orig_model.model.model, get_details=True)
        
        print(f"\nOriginal {orig_model.__class__.__name__} size:")
        print(f"Total Size: {original_size:.2f} MB")
        print('--------------------------------------')
        
        quantized_sam = copy.deepcopy(orig_model.model.model)
        print(f"\nQuantizing {orig_model.__class__.__name__}  model...")
        print(f"Quantize percentage: {pec}")
        del orig_model.model.model
        clean_memory()
    else: 
        # Original model size visualization
        original_size = calculate_model_size(orig_model.model, get_details=True)
        
        print(f"\nOriginal {orig_model.__class__.__name__} size:")
        print(f"Total Size: {original_size:.2f} MB")
        print('--------------------------------------')
        
        quantized_sam = copy.deepcopy(orig_model.model)
        
        print(f"\nQuantizing {orig_model.__class__.__name__} model...")
        print(f"Quantize percentage: {pec}.")
        del orig_model.model
        clean_memory()

    
        

    ##########
    # Quantize
    quantized_sam = quantize_sam_model(quantized_sam, quantization_percentage=pec, quant_type=quant_type)
    ##########

    # Replace the model with the quantized version
    if isinstance(orig_model, SAM2):
        orig_model.model.model = quantized_sam
        
        print('End Quantization')
        print('--------------------------------------')
        # Quantized model size visualization
        quantized_size = calculate_model_size(orig_model.model.model, get_details=True)
        print(f"\nQuantized {orig_model.__class__.__name__} size:")
        print(f"Total Size: {quantized_size:.2f} MB")
        print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.2f}%")
        
    else:
        orig_model.model = quantized_sam

        print('End Quantization')
        print('--------------------------------------')
        # Quantized model size visualization
        quantized_size = calculate_model_size(orig_model.model, get_details=True)
        print(f"\nQuantized {orig_model.__class__.__name__} size:")
        print(f"Total Size: {quantized_size:.2f} MB")
        print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.2f}%")


    






def clean_memory():
    """Clean up memory by garbage collecting and clearing CUDA cache if available."""
    # Force garbage collection
    gc.collect()
    
    # Clear PyTorch's CUDA cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Clear any unused memory pools
    if hasattr(torch.cuda, 'memory_allocated'):
        torch.cuda.memory_allocated(device=None)



# Check the current memory usage of CUDA
def check_cuda_memory():
    # Total memory available on GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory

    # Memory currently allocated
    allocated_memory = torch.cuda.memory_allocated(0)

    # Memory currently cached by PyTorch
    cached_memory = torch.cuda.memory_reserved(0)

    print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
    print(f"Memory allocated: {allocated_memory / 1e9:.2f} GB")
    print(f"Memory cached: {cached_memory / 1e9:.2f} GB")




def print_memory_usage():
    """Print current GPU memory usage."""
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")




def get_layers_to_quantize(model, percentage):
    """
    Returns a set of layer IDs that should be quantized.
    """
    # Get all Linear layers
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
    
    # Calculate number of layers to quantize
    n_layers = len(linear_layers)
    n_to_quantize = int(n_layers * (percentage / 100))
    
    # Randomly select layers
    layers_to_quantize = set(random.sample(linear_layers, n_to_quantize))
    
    print(f"Found {n_layers} Linear layers, will quantize {n_to_quantize} layers")
    return layers_to_quantize




def replace_linear_layer(module, quant_type):
    """
    Convert a single linear layer to 4-bit.
    quant_type: "nf4", or "fp4" ,or "int8"
    """
    quantized = bnb.nn.Linear4bit(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type=quant_type
    )
    # Copy the weights
    quantized.weight = bnb.nn.Params4bit(
        module.weight.data,
        requires_grad=False,
        quant_type=quant_type
    )
    if module.bias is not None:
        quantized.bias = torch.nn.Parameter(module.bias.data)
    return quantized




def quantize_sam_model(model, quantization_percentage, quant_type="nf4"):
    """Quantize the SAM model with specified percentage of layers."""
    # print("Initial memory usage:")
    # print_memory_usage()
    
    # Move model to CPU temporarily
    model = model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get layers to quantize
    layers_to_quantize = get_layers_to_quantize(model, quantization_percentage)
    
    # Quantize selected layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in layers_to_quantize:
            # print(f"Quantizing layer: {name}")
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                try:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, replace_linear_layer(module, quant_type))
                except AttributeError:
                    continue
    
    # Move back to GPU
    model = model.cuda()
    
    # print("\nFinal memory usage:")
    # print_memory_usage()
    
    return model




def verify_quantization(model):
    """Verify the quantization results."""
    linear_layers = 0
    quantized_layers = 0
    
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit)):
            linear_layers += 1
            if isinstance(module, bnb.nn.Linear4bit):
                quantized_layers += 1
    
    print(f"\nQuantization Results:")
    print(f"Total linear layers: {linear_layers}")
    print(f"Quantized layers: {quantized_layers}")
    print(f"Actual quantization rate: {(quantized_layers/linear_layers*100):.2f}%")



    
def calculate_model_size(model, get_details=False):
    """
    Calculate model size based on parameter count and data type.
    Returns size in MB.
    """
    def get_param_size(param):
        if isinstance(param, bnb.nn.Params4bit):
            return 4/8  # 4-bit = 0.5 bytes
        elif param.dtype == torch.float16:
            return 16/8  # float16 = 2 bytes
        elif param.dtype == torch.float32:
            return 32/8  # float32 = 4 bytes
        elif param.dtype == torch.int8:
            return 1  # int8 = 1 byte
        else:
            return 4  # default to float32 = 4 bytes

    total_params = 0
    total_size = 0
    size_dict = {
        '4bit': 0,
        'float16': 0,
        'float32': 0,
        'int8': 0,
        'other': 0
    }

    for name, param in model.named_parameters():
        num_params = param.numel()
        param_size = get_param_size(param)
        size_bytes = num_params * param_size
        
        total_params += num_params
        total_size += size_bytes

        # Track sizes by data type
        if isinstance(param, bnb.nn.Params4bit):
            size_dict['4bit'] += size_bytes
        elif param.dtype == torch.float16:
            size_dict['float16'] += size_bytes
        elif param.dtype == torch.float32:
            size_dict['float32'] += size_bytes
        elif param.dtype == torch.int8:
            size_dict['int8'] += size_bytes
        else:
            size_dict['other'] += size_bytes

    total_size_mb = total_size / (1024 * 1024)  # Convert to MB

    if get_details:
        print(f"\nModel Size Details:")
        print(f"Total Parameters: {total_params:,}")
        print(f"\nSize by data type (MB):")
        for dtype, size in size_dict.items():
            if size > 0:
                print(f"{dtype}: {size/(1024*1024):.2f} MB")
        return total_size_mb
    
    return total_size_mb



