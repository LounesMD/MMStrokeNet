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

## Methods and Materials

### Datasets

To ensure reproducibility, we provide detailed descriptions of the datasets used in this study. These include a publicly available single-modality dataset (ATLAS v2.0) and an internal dual-modality dataset, combining T1-weighted (T1-w) and FLAIR MRI scans.

#### 1. Public Single-Modality ATLAS v2.0 Dataset

The **ATLAS v2.0 dataset** was used to develop the baseline single-modality model. It includes T1-w MRI scans and corresponding lesion segmentation masks. This dataset is publicly available and pre-aligned to the MNI-152 standard template with a voxel size of 1 x 1 x 1 mm. The dataset consists of:

- **Training set**: 655 T1-w MRI scans with lesion masks (used for training and validating our model).
- **Test set**: 300 T1-w MRI scans. 
- **Hidden test set**: 316 T1-w MRI scans.

#### 2. Internal Dual-Modality Datasets

The **internal dual-modality datasets** were collected from two clinical studies: the **NeuroFB-AVC study** (NCT03766113) and the **AVCPOSTIM study** (NCT01677091). These datasets, which include both T1-weighted (T1-w) and FLAIR MRI scans, were used for fine-tuning the baseline model with dual-modality inputs, as well as for testing. Manual segmentation of the FLAIR images was carried out by a neuroimaging expert, with assistance from the T1-w modality, and reviewed by a neuroradiologist.

### Data Access

- The **ATLAS v2.0 dataset** is publicly available for download [here](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html).
- For access to the **internal datasets**, please contact the authors or refer to the corresponding clinical study registrations:
  - [NeuroFB-AVC Study (NCT03766113)](https://clinicaltrials.gov/ct2/show/NCT03766113)
  - [AVCPOSTIM Study (NCT01677091)](https://clinicaltrials.gov/ct2/show/NCT01677091)

### Ethical Considerations

All subjects in the internal datasets provided written informed consent before participation. The studies were approved by the relevant ethics committees and complied with French data confidentiality regulations.

---
## Preprocessing Pipeline

The preprocessing pipeline is a critical component in the preparation of MRI data prior to segmentation model training and testing. These preprocessing steps were applied to all MRI data involved in the project, including both the internal datasets and the ATLAS dataset. When running the model in test mode, you should ideally perform the preprocessing steps (1-5) on your data first.

### Preprocessing Steps

#### 1. **Brain Extraction**
The **HD-BET** tool is used to remove the skull from the images. This deep learning-based method provides improved accuracy for brain extraction compared to traditional methods. For more information on HD-BET, please refer to https://github.com/MIC-DKFZ/HD-BET.

#### 2. **Re-orientation**
The volumes are re-oriented to the **RAS** (Right-Anterior-Superior) coordinates to ensure consistent orientation across all images. This step is crucial for standardizing image orientation and preventing issues during further processing.

#### 3. **Registration**
If both **T1-w** and **FLAIR** images are available for a subject, the T1-w image is rigidly registered to the corresponding FLAIR image using a block matching registration method (**animaPyramidalBMRegistration**, **Anima** (https://anima.irisa.fr/)). If only the T1-w modality is available, this step is skipped. This ensures alignment between T1-w and FLAIR images when both are present, supporting multi-modality analysis.

#### 4. **Bias Correction**
The bias due to spatial inhomogeneity is estimated and removed from the data using the **N4ITK** bias field correction algorithm (**animaN4BiasCorrection**). This correction compensates for signal variations caused by magnetic field inhomogeneities during MRI acquisition, ensuring more uniform intensity across the image.

#### 5. **Intensity Normalization**
Image intensities are standardized by subtracting the mean voxel value and dividing by the standard deviation for each image. This normalization step ensures that all images have a consistent intensity distribution, making them comparable across subjects and modalities, which is crucial for downstream analyses and model training.

#### Note on nnU-Net Framework

It is important to note that in addition to the preprocessing steps outlined above, the **nnU-Net** framework (**https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1**) performs several additional operations by default to standardize the input data:

- **Voxel Spacing Standardization**: The framework automatically calculates a uniform target spacing for each axis using the median values from the training dataset. This ensures that all images are resampled to have consistent voxel spacing across subjects and modalities.
  
- **Resampling Strategy**: By default, **third-order spline interpolation** is used for resampling images. However, for anisotropic images (where the ratio of the maximum to minimum axis spacing exceeds 3), the framework adapts its resampling approach:
  - **In-plane resampling** is done using third-order spline interpolation.
  - **Out-of-plane resampling** uses **nearest neighbor interpolation** to reduce resampling artifacts and better preserve spatial information.

- **Segmentation Maps**: The framework automatically converts segmentation maps into **one-hot encodings**. Each channel of the segmentation mask is then interpolated using **linear interpolation**, and the final segmentation mask is obtained by applying the **argmax** operation. For anisotropic data, **nearest neighbor interpolation** is applied to the low-resolution axis to minimize interpolation artifacts.

This preprocessing ensures consistency across all images and improves the robustness of the model, especially when dealing with different image resolutions and anisotropic data.

## Test Mode Example

To run the code in test mode, you can use the following two command examples. These commands are designed to predict on a batch of data and evaluate the models based on the given inputs.

Note: you need to perform step 1 of the preproessing pipeline first.

This command performs preprocessing steps (2-5) and generates predictions on test data:

```bash
python3 preprocess.py --patients <input_patient_folder>/raw/ --preprocess_steps prepare_without_brain_extraction remove_bias normalize -cs -m t1 flair

python3 predict_batch.py --patients <input_patient_folder>/normalized/ --output <output_folder> -cs -mn <model_name> -m <modality> --preprocess_steps check install -pmin <min_value> -pmax <max_value> -vmin <validation_min> -cfg <config_file> -f <fold_number>

python3 evaluate_models.py -mo <model_output_folder> -gt <input_patient_folder>/raw/ -o <evaluation_results_folder> -cs -psigts -sgp -cfg <config_file>
```
In our study we used: 
-pmin 0.2 -pmax 0.2 -vmin 10

The evaluation script uses **animaSegPerfAnalyzer** for lesion detection performance analysis.

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
The original research paper is currently under review.
Initial results were presented in the form of an oral presentation at the 13th World Congress for Neurorehabilitation (WCNR) 2024. Here is a [link](https://hal.science/hal-04546362) to the paper's abstract.

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
