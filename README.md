## Neuroscience project

- [fMRI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC162295/) detects the blood oxygen level–dependent (BOLD) changes in the MRI signal that arise when changes in neuronal activity occur following a change in brain state, such as may be produced, for example, by a stimulus or task.
- Population receptive field ([pRF](https://brainlife.io/docs/tutorial/prf-mapping/)) modeling is a popular fMRI method to map the retinotopic organization of the human brain.
- [fLoc](https://github.com/VPNL/fLoc) - The use of fMRI scans to first identify brain areas of functional significance and then more closely examine the brain activity in the area is popular in neuroimaging and is known as [functional localization](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9167083/). This approach requires conducting an independent experiment to localize the targeted region for each individual participant.
  

### NSD
  - [Natural scenes Dataset](https://osf.io/zyb3t/wiki/home/) - also contains github link and supplementary videos
  - [Supplementary info](https://www.nature.com/articles/s41593-021-00962-x#Sec55)
  - [Video of raw fMRI scan of subj01](https://osf.io/5sx2p)
  - [Download data using AWS CLI](https://aws.amazon.com/marketplace/pp/prodview-otyj4ovtx7ypo#resources)
  - [Browse the data](https://natural-scenes-dataset.s3.amazonaws.com/index.html)

### Algonaut 2023 challenge
  - Predict neural responses to visual stimuli using Encoding models
  - Encoding model typically consists of an algorithm that takes image pixels as input, transforms them into model features, and maps these features into brain data(e.g. fMRI activity), effectively predicting the neural responses to images.
  - Data
      - 8 subjects
      - Train split - [9841, 9841, 9082, 8779, 9841, 9082, 9841, 8779] images
      - Test split - [159, 159, 293, 395, 159, 293, 159, 395] images
      - ROI indices for selecting vertices belonging to specific visual ROIs are provided
      - The fMRI data is z-scored(mean - 0, variance - 1) within each NSD scan session and averaged across image repeats
      - The visual cortex is divided into multiple areas having different functional properties, referred to as regions-of-interest (ROIs).
          - **Early retinotopic visual regions (prf-visualrois):** V1v, V1d, V2v, V2d, V3v, V3d, hV4.
          - **Body-selective regions (floc-bodies):** EBA, FBA-1, FBA-2, mTL-bodies.
          - **Face-selective regions (floc-faces):** OFA, FFA-1, FFA-2, mTL-faces, aTL-faces.
          - **Place-selective regions (floc-places):** OPA, PPA, RSC.
          - **Word-selective regions (floc-words):** OWFA, VWFA-1, VWFA-2, mfs-words, mTL-words.
          - **Anatomical streams (streams):** early, midventral, midlateral, midparietal, ventral, lateral, parietal.
  - Baseline
      - the baseline model score of the challenge reflects a linearizing encoding model built using a **pretrained AlexNet**. Its mean noise-normalized prediction accuracy over all subjects, hemispheres and vertices is **40.48%** of the total predictable variance.
        
  - [Colab tutorial](https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link)
  - [Paper](https://arxiv.org/abs/2301.03198)
        
### Detailed Model summary
  - You can use [torchinfo](https://github.com/tyleryep/torchinfo) package for this purpose
    ```
    import torch
    from torchinfo import summary
    res_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    summary(res_model, col_names=["input_size", "output_size", "num_params", "kernel_size"], input_size=(1, 3, 224, 224))
    ```
    <img width="929" alt="Screenshot 2023-07-18 at 3 00 53 PM" src="https://github.com/mkmohan003/gallirallus/assets/134788080/e67816cc-21d7-4702-a306-d7e6b8ec677b">

  

    
