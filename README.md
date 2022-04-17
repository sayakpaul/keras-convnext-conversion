This repository holds the code that was used to populate the official ConvNeXt
parameters [1, 2] into Keras ConvNeXt implementation. Most of the code is copied
from here: https://github.com/sayakpaul/ConvNeXt-TF. Please refer to this repository
for more comments, setup guides, etc. 

The conversion was performed to aid this PR: https://github.com/keras-team/keras/pull/16421.

## Execution

1. Install the Python dependencies: `pip install -r requirements.txt`.
2. Make sure you're at the root of the repository after cloning it.
3. Then create the required directories:

  ```sh
  $ mkdir keras-applications
  $ mkdir keras-applications/convnext
  ```
4. Then execute: `python convert_all.py`.

## References

[1] ConvNeXt paper: https://arxiv.org/abs/2201.03545

[2] Official ConvNeXt code: https://github.com/facebookresearch/ConvNeXt