## Neuroscience project

### NSD
  - [Natural scenes Dataset](https://osf.io/zyb3t/wiki/home/) - also contains github link and supplementary videos
  - [Supplementary info](https://www.nature.com/articles/s41593-021-00962-x#Sec55)
  - [Video of raw fMRI scan of subj01](https://osf.io/5sx2p)

### Algonaut 2023 challenge
  - Predict neural responses to visual stimuli using Encoding models
  - Encoding model typically consists of an algorithm that takes image pixels as input, transforms them into model features, and maps these features into brain data(e.g. fMRI activity), effectively predicting the neural responses to images.
  - Data
      - 8 subjects
      - Train split - [9841, 9841, 9082, 8779, 9841, 9082, 9841, 8779] images
      - Test split - [159, 159, 293, 395, 159, 293, 159, 395] images
      - ROI indices for selecting vertices belonging to specific visual ROIs are provided
      - The fMRI data is z-scored(mean - 0, variance - 1) within each NSD scan session and averaged across image repeats
  - [Colab tutorial](https://colab.research.google.com/drive/1bLJGP3bAo_hAOwZPHpiSHKlt97X9xsUw?usp=share_link)
  - [Paper](https://arxiv.org/abs/2301.03198)
        

