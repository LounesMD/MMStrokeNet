# Deep Learning and Multi-Modal MRI for Segmentation of Sub-Acute and Chronic Stroke Lesions

This repository provides the implementation of the MM-StrokeNet method for segmenting sub-acute and chronic stroke lesions using T1-w and FLAIR MRI modalities.

<!-- 
<p align=center>
  <img src="./Images/Gif-Seg2.gif" width="400" height="350">
  <img src="./Images/Gif-Seg.gif" width="400">
</p>-->

Main contacts : 

Lounès Meddahi lounes.meddahi@ens-rennes.fr  
Francesca Galassi francesca.galassi@irisa.fr

## Overview

This repository contains the code for fine-tuning a pre-trained single-modality nnU-Net model to handle two modalities (T1-w and FLAIR MRIs), as well as the following trained models: the baseline single-modality model trained on the ATLAS v2.0 dataset, the fine-tuned single-modality T1-w model, and the fine-tuned dual-modality T1-w + FLAIR model. Fine-tuning was performed on a private dataset. The entire pipeline, the adaptation process, and the models are described in our paper "Deep Learning and Multi-Modal MRI for Segmentation of Sub-Acute and Chronic Stroke Lesions", currently under review.

<!--
<p align=center>
  <img src="./Images/ImageGit.svg" width="700" title="Model_v1 trained on ATLAS T1-weighted MRIs">
</p>-->

## Preprocessing Pipeline

Our approach builds upon our previously proposed framework [18, 21], with adjustments to the preprocessing pipeline aimed at improving brain extraction. Specifically, we replaced the previous brain extraction step, which used **Anima** (https://anima.irisa.fr/), with the state-of-the-art HD-BET deep learning-based tool [22]. Below is a description of the preprocessing pipeline:

### Preprocessing Steps

#### 1. **Brain Extraction**
The **HD-BET** tool is used to remove the skull from the images. This deep learning-based method provides improved accuracy for brain extraction compared to traditional methods. For more information on HD-BET, please refer to [the HD-BET repository](https://github.com/MIC-DKFZ/HD-BET).

#### 2. **Re-orientation**
The volumes are re-oriented to the **RAS** (Right-Anterior-Superior) coordinates to ensure consistent orientation across all images. This step is crucial for standardizing image orientation and preventing issues during further processing.

#### 3. **Registration**
If both **T1-weighted** (T1-w) and **FLAIR** images are available for a subject, the T1-w image is rigidly registered to the corresponding FLAIR image using a block matching registration method (**animaPyramidalBMRegistration**). If only the T1-w modality is available, this step is skipped. This ensures alignment between T1-w and FLAIR images when both are present, supporting multi-modality analysis.

#### 4. **Bias Correction**
The bias due to spatial inhomogeneity is estimated and removed from the data using the **N4ITK** bias field correction algorithm. This correction compensates for signal variations caused by magnetic field inhomogeneities during MRI acquisition, ensuring more uniform intensity across the image.

#### 5. **Intensity Normalization**
Image intensities are standardized by subtracting the mean voxel value and dividing by the standard deviation for each image. This normalization step ensures that all images have a consistent intensity distribution, making them comparable across subjects and modalities, which is crucial for downstream analyses and model training.

## Installation and Requirements

This section provides a step-by-step guide on how to install and run MM-StrokeNet. Before proceeding, ensure your system meets the following requirements:

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
Clone this repository with the following command:

```bash
git clone https://github.com/LounesMD/MM_StrokeNet.git


## MM-StrokeNet packages
In this repository you will find the following folders :
* Algorithm : Custom scripts used to modify the pretrained nnU-Net model for compatibility with dual-modality (T1-w + FLAIR) inputs.
* Images : Visual assets used in the paper, including images and figures.
* Models : Pretrained and fine-tuned model weights and configurations. This includes:
	*   The baseline model trained on the [ATLAS v2.0 dataset](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html).
	*   The fine-tuned model on T1-weighted MRIs.
	*   The fine-tuned model on both T1-weighted and FLAIR MRIs.

## Paper :
The original research paper is currently under review. Initial results were presented in the form of an oral presentation at the 13th World Congress for Neurorehabilitation (WCNR) 2024. Here is a [link](https://hal.science/hal-04546362) to the paper's abstract.

## Citing 
If you use this project in your work, please consider citing it as follows:

```bibtex
@misc{MM-STROKEnet,
  authors = {Lounès Meddahi, Stéphanie s Leplaideur, Arthur Masson, Isabelle Bonan, Elise Bannier Francesca Galassi},
  title = {Enhancing stroke lesion detection and segmentation through nnU-net and multi-modal MRI Analysis},
  year = {2024},
  conference = {WCNR 2024 - 13th World Congress for Neurorehabilitation, World federation for Neurorehabilitation},
}
```
