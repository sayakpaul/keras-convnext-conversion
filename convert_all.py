import os

from tqdm import tqdm

"""
Details about these checkpoints are available here:
https://github.com/facebookresearch/ConvNeXt#results-and-pre-trained-models.
"""

imagenet_1k_224 = {
  "convnext_tiny": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
  "convnext_small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
  "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth",
  "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth",
  "convnext_xlarge": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth",
}

print("Converting 224x224 resolution ImageNet-1k models.")
for model in tqdm(imagenet_1k_224):
    print(f"Converting {model} with classification top.")
    command_top = f"python convert.py -m {model} -c {imagenet_1k_224[model]} -t"
    os.system(command_top)

    print(f"Converting {model} without classification top.")
    command_no_top = f"python convert.py -m {model} -c {imagenet_1k_224[model]}"
    os.system(command_no_top)
