import torch
import sys

model = torch.load('Real-ESRGAN/weights/realesr-general-x4v3.pth', map_location='cpu')
params = model['params']

print("Keys in model file:", list(model.keys()))
print(f"Total parameters: {len(params.keys())}")

# Print the first 20 parameters with their shapes
print("\nFirst 20 parameters:")
for i, key in enumerate(sorted(params.keys())):
    print(f"{key:30s} - {params[key].shape}")
    if i >= 20:
        break

# Print some specific parameters to understand their structure
for name in ['body.0.weight', 'body.1.weight', 'body.2.weight']:
    if name in params:
        print(f"\n{name} shape: {params[name].shape}")

print("\nExamining parameter patterns...")
# Check body parameters pattern
body_weights = [k for k in params.keys() if k.startswith('body.') and '.weight' in k]
print(f"Total body weights: {len(body_weights)}")
if body_weights:
    print("Sample body weights:")
    for i in range(min(5, len(body_weights))):
        print(f"  {body_weights[i]} - {params[body_weights[i]].shape}")

# Check for conv_first, upconv, etc.
conv_names = ['conv_first', 'conv_hr', 'conv_last', 'upconv1', 'upconv2']
for name in conv_names:
    weight_key = f"{name}.weight"
    bias_key = f"{name}.bias"
    if weight_key in params:
        print(f"{weight_key:30s} - {params[weight_key].shape}")
    if bias_key in params:
        print(f"{bias_key:30s} - {params[bias_key].shape}")
