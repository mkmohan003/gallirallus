#!/usr/bin/env python3

'''
Right now pile of functions to simplify NBs derived from the 
NSD algonauts tutorial NB.
'''

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr

def extract_features(feature_extractor, dataloader, pca):
    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
    return np.vstack(features)


def extract_features_train_val_test(extractor, dataloaders: tuple, pca):
  train_dataloader, val_dataloader, test_dataloader = dataloaders

  ftrain = extract_features(extractor, train_dataloader, pca)
  fval = extract_features(extractor, val_dataloader, pca)
  ftest = extract_features(extractor, test_dataloader, pca)

  return ftrain, fval, ftest


def fit_mapping_correlations(ftrain, fval, ftest, lh_fmri_train, rh_fmri_train,
                             lh_fmri_val, rh_fmri_val):
  # Fit linear regressions on the training data
  reg_lh = LinearRegression().fit(ftrain, lh_fmri_train)
  reg_rh = LinearRegression().fit(ftrain, rh_fmri_train)
  # Use fitted linear regressions to predict the validation and test fMRI data
  lh_fmri_val_pred = reg_lh.predict(fval)
  lh_fmri_test_pred = reg_lh.predict(ftest)
  rh_fmri_val_pred = reg_rh.predict(fval)
  rh_fmri_test_pred = reg_rh.predict(ftest)

  # Empty correlation array of shape: (LH vertices)
  lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
  # Correlate each predicted LH vertex with the corresponding ground truth vertex
  for v in tqdm(range(lh_fmri_val_pred.shape[1])):
      lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

  # Empty correlation array of shape: (RH vertices)
  rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
  # Correlate each predicted RH vertex with the corresponding ground truth vertex
  for v in tqdm(range(rh_fmri_val_pred.shape[1])):
    rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]
  return lh_correlation, rh_correlation


def create_plot(lh_correlation, rh_correlation, data_dir):
  # Load the ROI classes mapping dictionaries
  roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
      'mapping_floc-faces.npy', 'mapping_floc-places.npy',
      'mapping_floc-words.npy', 'mapping_streams.npy']
  roi_name_maps = []
  for r in roi_mapping_files:
      roi_name_maps.append(np.load(os.path.join(data_dir, 'roi_masks', r),
          allow_pickle=True).item())

  # Load the ROI brain surface maps
  lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
      'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
      'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
      'lh.streams_challenge_space.npy']
  rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
      'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
      'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
      'rh.streams_challenge_space.npy']
  lh_challenge_rois = []
  rh_challenge_rois = []
  for r in range(len(lh_challenge_roi_files)):
      lh_challenge_rois.append(np.load(os.path.join(data_dir, 'roi_masks',
          lh_challenge_roi_files[r])))
      rh_challenge_rois.append(np.load(os.path.join(data_dir, 'roi_masks',
          rh_challenge_roi_files[r])))

  # Select the correlation results vertices of each ROI
  roi_names = []
  lh_roi_correlation = []
  rh_roi_correlation = []
  for r1 in range(len(lh_challenge_rois)):
      for r2 in roi_name_maps[r1].items():
          if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
              roi_names.append(r2[1])
              lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
              rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
              lh_roi_correlation.append(lh_correlation[lh_roi_idx])
              rh_roi_correlation.append(rh_correlation[rh_roi_idx])
  roi_names.append('All vertices')
  lh_roi_correlation.append(lh_correlation)
  rh_roi_correlation.append(rh_correlation)

  # Create the plot
  lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r])
      for r in range(len(lh_roi_correlation))]
  rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r])
      for r in range(len(rh_roi_correlation))]
  plt.figure(figsize=(18,6))
  x = np.arange(len(roi_names))
  width = 0.30
  plt.bar(x - width/2, lh_mean_roi_correlation, width, label='Left Hemisphere')
  plt.bar(x + width/2, rh_mean_roi_correlation, width,
      label='Right Hemishpere')
  plt.xlim(left=min(x)-.5, right=max(x)+.5)
  plt.ylim(bottom=0, top=1)
  plt.xlabel('ROIs')
  plt.xticks(ticks=x, labels=roi_names, rotation=60)
  plt.ylabel('Mean Pearson\'s $r$')
  plt.legend(frameon=True, loc=1)
  return lh_mean_roi_correlation, rh_mean_roi_correlation


####################
### fMRI related ###
####################

def get_roi_class(roi):
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = 'prf-visualrois'
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = 'floc-bodies'
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = 'floc-faces'
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = 'floc-places'
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = 'floc-words'
    elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
        roi_class = 'streams'
    return roi_class