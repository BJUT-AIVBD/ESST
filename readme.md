# When zero-padding position encoding encounters linearspace reduction attention: An efficient semantic segmentation Transformer ofremote sensing images

## Installation

conda env create -n esst python=3.8
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install mmcv-full==1.3.0
cd to/root/path
pip install -e .

### Train on Potsdam and Vaihingen dataset

 For example, when dataset is Potsdam and method is ESST, we can run

python tools/train.py \
  --config configs/lightformer/mlphead_lightformer_potsdam_160k.py\
  --work-dir result/ESST \
  --load_from path/to/pre-trained/model \

### Inference on Potsdam and Vaihingen dataset

For example, when dataset is Potsdam and method is ESST, we can run

python tools/test.py \
  --config configs/lightformer/mlphead_lightformer_potsdam_160k.py \
  --checkpoint path/to/ESST/model \
  --show_dir result/ESST/test \

## Hyperparameters Configuration

Detailed hyperparameters config can be found in configs/_base_/

## Acknowledgments

The code is developed based on the following repositories. We appreciate their nice implementations.

### Method Repository

Swin Transformer    https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

UPerNet    https://github.com/CSAILVision/unifiedparsing

CSwin    https://github.com/microsoft/CSWin-Transformer

PVT/PVT v2    https://gitcode.com/mirrors/whai362/pvt

### Cite this repository

If you use this software in your work, please cite it using the following metadata. Yi Yan, Jing Zhang, Xinjia Wu, Jiafeng Li, and Li Zhuo. (2023). ESST by BJUT-AI&VBD [Computer software]. https://github.com/BJUT-AIVBD/ESST
