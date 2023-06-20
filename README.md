# MM-STROKEnet: A Novel Approach for Stroke Lesion Segmentation Using multi MRI Modalities
This package provides an implementation of the MM-StrokeNet method. This is a new model for stroke lesions segmentation from T1+FLAIR modalities.

<p align=center>
  <img src="./Images/Gif-Seg2.gif" width="400" height="350">
  <img src="./Images/Gif-Seg.gif" width="400">
</p>
Main contact : Lounès Meddahi (lounes.meddahi@ens-rennes.fr)


## Overview
This repository contains the code and models used for brain stroke segmentation using for our MM-Strokenet using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)/[Longiseg4MS](https://gitlab.inria.fr/amasson/longiseg4ms). The model was first trained on T1-weighted MRIs from the [ATLAS v2.0 dataset](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html), and then adjusted using a custom [algorithm](./Algorithm/finetune_Script.py) to make it compatible with T1+FLAIR MRIs. It was then further finetuned using T1-weighted+FLAIR MRIs from our own in-house dataset (not publicly available for the moment). The repository also contains the weights and results from [both models](./Models/).

<p align=center>
  <img src="./Images/ImageGit.svg" width="700" title="Model_v1 trained on ATLAS T1-weighted MRIs">
</p>

## Installation and Requirements

This section provides a step-by-step guide on how to install and run STROKEnet. Before proceeding, ensure your system meets the following requirements:

### Software Requirements
- Python 3.9 or higher
- Torch 2.0.0 or higher
- Scipy
- NumPy
- scikit-learn
- scikit-image 0.19.3 or higher

### Hardware Used
- NVIDIA GPU (RTX A5000 GPU and 30,6GiB RAM) with CUDA 11.4

### Installation Steps
Clone this repository with the following command :

```git clone https://github.com/LounesMD/MM_StrokeNet.git```

## MM-StrokeNet packages
In this repository you will find the following folders :
* Algorithm : Our custom algorithm is designed to modify a pretrained model, allowing it to conform to the architecture of a different model, thereby facilitating the transfer of learned features between models.
* Images : All visual assets used in the paper, such as images and figures.
* Models : The weights and models for our pretrained and finetuned networks

If you have any question about the source code, please contact me.

## Paper :
If you'd like to take a look at the preprint, please feel free to do so : [preprint](https://github.com/LounesMD/MMStrokeNet/blob/main/Preprint_MMStrokeNet.pdf).

## Citing 
If you use the project in your work, please consider citing it with:

```bibtex
@misc{MM-STROKEnet,
  author = {Lounès Meddahi, Arthur Masson, Elise Bannier, Stéphanie Leplaideur, Francesca Galassi},
  title = {Enhancing Chronic Stroke Lesion Detection and Segmentation through nnU-net and Multi-Modal MRI Analysis},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LounesMD/MMStrokeNet}},
}
```
