"""
Simple, clean DeOldify colorization script without any fallbacks.
If colorization fails, it raises an exception rather than silently using a fallback.
"""
import os
import sys
import torch
import argparse
from PIL import Image
from pathlib import Path

# Add DeOldify to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
deoldify_dir = os.path.abspath(os.path.join(repo_root, 'DeOldify'))
if deoldify_dir not in sys.path:
    sys.path.insert(0, deoldify_dir)

# Fix torch.load for PyTorch 2.6
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """Patch torch.load to always use weights_only=False for PyTorch 2.6 compatibility"""
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load

# Import DeOldify after patching torch.load
try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    from deoldify.visualize import get_image_colorizer
except ImportError:
    raise ImportError("DeOldify not found! Make sure the DeOldify directory exists at the correct location.")

def monkey_patch_fastai():
    """Monkey patch fastai's load_learner to use custom model directory"""
    import fastai
    import torch
    from fastai.basic_train import load_learner

    # Store original function
    original_load = fastai.basic_train.load_learner

    # Path to models directory
    model_dir = os.path.abspath(os.path.join(repo_root, 'My PTH'))
    print(f"Setting up model directory: {model_dir}")

    # Also patch torch.load to always use weights_only=False
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        print(f"Patched torch.load called with path: {args[0] if args else 'unknown'}")
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    
    # Now patch fastai loader
    def patched_load_learner(*args, **kwargs):
        print(f"Patched load_learner called")
        
        # Replace any path argument containing 'models'
        if 'path' in kwargs and 'models' in str(kwargs['path']):
            print(f"Replacing path: {kwargs['path']} -> {model_dir}")
            kwargs['path'] = model_dir
        elif len(args) > 0 and 'models' in str(args[0]):
            print(f"Replacing args[0]: {args[0]} -> {model_dir}")
            args = list(args)
            args[0] = model_dir
            args = tuple(args)
            
        # Also check if the original path being requested exists
        if len(args) > 0 and isinstance(args[0], str) and not os.path.exists(args[0]):
            print(f"WARNING: Path does not exist: {args[0]}")
            if os.path.exists(model_dir):
                print(f"Using model dir instead: {model_dir}")
                args = list(args)
                args[0] = model_dir
                args = tuple(args)
        
        # Apply additional patches to fix paths
        import fastai.torch_core
        fastai.torch_core.defaults.device = torch.device('cpu')
        
        # Fix model paths in the fastai library
        for sub_module in [fastai.basic_data, fastai.basic_train, fastai.vision.data, fastai.vision.learner]:
            if hasattr(sub_module, 'models_path'):
                sub_module.models_path = lambda: model_dir
        
        return original_load(*args, **kwargs)
    
    # Apply the patch
    fastai.basic_train.load_learner = patched_load_learner
    
    # Also patch the torch.load calls in learn.load
    original_learn_load = fastai.basic_train.Learner.load
    
    def patched_learn_load(self, name, device=None, strict=True, with_opt=None, purge=True, **kwargs):
        print(f"Patched Learner.load called with name: {name}")
        # Force non-weights-only loading
        original_torch_load_ref = torch.load
        torch.load = patched_torch_load
        
        # Check if we're loading from a models directory
        if 'models' in str(name):
            name = name.replace('models/', '')
            name = name.replace('models\\', '')
            name = os.path.join(model_dir, os.path.basename(name))
            print(f"Updated model path: {name}")
        
        result = original_learn_load(self, name, device, strict, with_opt, purge, **kwargs)
        # Restore torch.load just in case
        torch.load = original_torch_load_ref
        return result
    
    fastai.basic_train.Learner.load = patched_learn_load
    
    print("Monkey patching complete for all model loading functions")

def colorize_image(input_path, output_path, model='stable', render_factor=25):
    """
    Colorize a single image using DeOldify without any fallbacks.
    
    Args:
        input_path: Path to input image file
        output_path: Path to save colorized output
        model: 'stable' or 'artistic'
        render_factor: Render factor (higher = better quality but slower)
        
    Returns:
        Path to colorized image if successful
        
    Raises:
        RuntimeError if colorization fails
    """
    print(f"\n======== COLORIZING IMAGE ========")
    print(f"Input:         {input_path}")
    print(f"Output:        {output_path}")
    print(f"Model:         {model}")
    print(f"Render Factor: {render_factor}")
    
    # Validate the model parameter
    if model not in ['stable', 'artistic']:
        raise ValueError("Model must be either 'stable' or 'artistic'")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Force CPU mode
    print("Setting DeOldify to use CPU...")
    device.set(device=DeviceId.CPU)
    
    # Apply the monkey patch for fastai
    monkey_patch_fastai()
    
    # Set model environment variable
    model_dir = os.path.abspath(os.path.join(repo_root, 'My PTH'))
    os.environ['DEOLDIFY_MODELS'] = model_dir
    print(f"Set DEOLDIFY_MODELS environment variable to: {model_dir}")
    
    # Verify model file exists
    if model == 'stable':
        model_path = os.path.join(model_dir, 'ColorizeStable_gen.pth')
    else:  # artistic
        model_path = os.path.join(model_dir, 'ColorizeArtistic_gen.pth')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Using model file: {model_path} (Exists: {os.path.exists(model_path)}, Size: {os.path.getsize(model_path)} bytes)")
    
    # Initialize colorizer
    try:
        print("Initializing colorizer...")
        colorizer = get_image_colorizer(artistic=(model == 'artistic'))
        
        # Process the image
        print(f"Colorizing image...")
        result_path = colorizer.plot_transformed_image(
            input_path,
            render_factor=render_factor,
            watermarked=False,
            post_process=True,
            results_dir=os.path.dirname(output_path),
            force_cpu=True  # Always use CPU
        )
        
        if not isinstance(result_path, str) or not os.path.exists(result_path):
            raise RuntimeError(f"Colorization failed. No output file was generated.")
            
        # Move result to expected output location if needed
        if os.path.abspath(result_path) != os.path.abspath(output_path):
            import shutil
            shutil.copy(result_path, output_path)
            print(f"Copied colorized image to final location: {output_path}")
            
        print(f"Colorization successful: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Colorization failed with error: {str(e)}")
        # Re-raise the exception instead of using a fallback
        raise RuntimeError(f"DeOldify colorization failed: {str(e)}")

def process_directory(input_dir, output_dir, model='stable', render_factor=25):
    """Process all images in a directory"""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing all images in {input_dir}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    success_count = 0
    failure_count = 0
    
    for file in os.listdir(input_dir):
        # Check if it's an image file
        if not any(file.lower().endswith(ext) for ext in image_extensions):
            continue
            
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"{Path(file).stem}_colorized.png")
        
        print(f"\nProcessing: {file}")
        try:
            colorize_image(input_path, output_path, model, render_factor)
            success_count += 1
        except Exception as e:
            print(f"FAILED: {file} - {str(e)}")
            failure_count += 1
            
    print(f"\nProcessing complete.")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    
    if failure_count > 0:
        print("WARNING: Some images failed to colorize. No fallbacks were used.")

def main():
    parser = argparse.ArgumentParser(description='Clean DeOldify colorization without fallbacks')
    parser.add_argument('--input', '-i', required=True, help='Input image file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output image file or directory')
    parser.add_argument('--model', '-m', choices=['stable', 'artistic'], default='stable',
                        help='Colorization model: stable or artistic (default: stable)')
    parser.add_argument('--render-factor', '-r', type=int, default=25,
                        help='Render factor (higher = better quality but slower) (default: 25)')
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        return 1
        
    # Process directory or single file
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.model, args.render_factor)
    else:
        try:
            colorize_image(args.input, args.output, args.model, args.render_factor)
            print(f"Success! Colorized image saved to: {args.output}")
        except Exception as e:
            print(f"ERROR: Colorization failed: {str(e)}")
            return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
