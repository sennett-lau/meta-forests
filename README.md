# Meta-forests

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Datasets](#datasets)
- [Citation](#citation)

## Description

This repository contains the code for this [paper](https://arxiv.org/abs/2401.04425).

## Installation

Setup a virtual environment with [Python 3.10](https://www.python.org/downloads/) and install [CUDA with version 12.6+](https://developer.nvidia.com/cuda-toolkit):
```bash
# sample command with conda virtual environment creation
conda create -n meta-forests python=3.10
conda activate meta-forests
```

Install the requirements:
```bash
pip install -r requirements.txt
```

Or install with command:
```bash
pip install deeplake==4.1.10
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> ⚠️ Please be reminded that Deeplake v4 does not support Windows, please proceed with WSL if you are on Windows.

## Datasets

We have used the following datasets according to the paper:
- [VLCS](https://github.com/belaalb/G2DM#download-vlcs)
- [PACS](https://domaingeneralization.github.io/#data)

While the [blood glucose monitoring dataset](https://ieeexplore.ieee.org/document/10181112) from Sun et al. (2023) used in the paper is not publicly available.

### Downloading the datasets

1. Download the VLCS dataset from the [link](http://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file).
2. Download the PACS dataset with the function `load_pacs_training_dataset()` in `src/load_data.py`.

## Citation

```
@misc{sun2024metaforests,
      title={Meta-forests: Domain generalization on random forests with meta-learning}, 
      author={Yuyang Sun, Panagiotis Kosmas},
      year={2024},
      eprint={2401.04425},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.04425}, 
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
