# SCARF
## Introduction

We propose a Semantic Constrained Attention ReFinement (SCARF) network. Our model can efficiently capture the long-range contextual information with semantic constraint layer-by-layer, which enhances the semantic information and structure reasoning of the model. We achieve superior performance on three challenging scene segmentation datasets, i.e., PASCAL VOC 2012, PASCAL Context and Cityscapes datasets.

## Usage

1. Install pytorch

   Our code is conducted on python 3.5 and torch 1.4.0.

2. Install environment

   clone the repository and open the folder, run

   ```
   python setup.py
   ```

   or

   ```
   pip install -e .
   ```

3. Dataset

   **PASCAL VOC 2012**: Download the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)  and [augmentation data](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0), then convert the dataset into trainaug, trainval, and test sets for training, fine-tuning and testing, respectively.

   **Cityscapes**: Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/) and convert the dataset to [19 categories](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).

   **PASCAL Context**: run script `scripts/prepare_pcontext.py`.

4. Training on PASCAL VOC 2012 dataset

   ```
   cd ./experiments/segmentation/
   ```

   Training on the trainaug set:

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_aug2 --backbone resnet101s --model scarf --checkname myname --ft --DS --dilated
   ```

   Fine-tuning on the trainval set:

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_voc --backbone resnet101s --model scarf --checkname myname --resume "pretrained_model_path" --ft --DS --dilated
   ```

   where `pretrained_model_path` is the path to the model trained on trainaug set.

5. Evaluation on PASCAL VOC 2012 dataset

   ```
   cd ./experiments/segmentation/
   ```

   Single scale testing on val set for model (trained on trainaug set):

   ```
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset pascal_voc --backbone resnet101s --model scarf --resume "pretrained_model_path" --eval --dilated
   ```

   One can download pretrained [SCARF model](https://pan.baidu.com/s/12jRtaiv-Vl7yjgyE0V3_KA)  with password 92b9 (trained on trainaug set) for easy testing. The expected score will show (mIoU/pAcc): 81.58/95.86.




