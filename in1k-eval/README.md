This directory provides a notebook and ImageNet-1k class mapping file to run 
evaluation on the ImageNet-1k `val` split using the TF/Keras converted ConvNeXt
models. The notebook assumes the following files are present in your working
directory and the dependencies specified in `../requirements.txt` are installed:

* The `val` split directory of ImageNet-1k.
* The class mapping files (`.json`).

The evaluation results can be found [here](https://tensorboard.dev/experiment/wGejlqbYRtGUKSJoi89asQ/#scalars).

## Comparison to the reported numbers

| name | original acc@1 | keras acc@1 |
|:---:|:---:|:---:|
| convnext_tiny_1k_224 | 82.1 | 81.312 |
| convnext_small_1k_224 | 83.1 | 82.392 |
| convnext_base_21k_1k_224 | 85.8 | 85.364 |
| convnext_large_21k_1k_224 | 86.6 | 86.36 |
| convnext_xlarge_21k_1k_224 | 87.0 | 86.732 |