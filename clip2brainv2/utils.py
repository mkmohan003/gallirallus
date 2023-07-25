import torch
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path


FEATURE_PATH = [
   'subj01_train_img.npy'
   'subj01_train_caption.npy'
]


def extract_store_features(model, dataloader, path):
    z_i_total = []
    z_t_total = []
    for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
      img, lh_fmri, rh_fmri, tokens = inputs
      with torch.no_grad():
        z_t = model.encode_text(tokens)
        z_i = model.encode_image(img) 

        z_i = z_i.detach().cpu().numpy()
        z_t = z_t.detach().cpu().numpy()

        z_i_total.append(z_i)
        z_t_total.append(z_t)


    z_i_total = np.concatenate(z_i_total, axis=0)
    z_t_total = np.concatenate(z_t_total, axis=0)

    with open(f'{path}_img.npy', 'wb') as f:
      np.save(f, z_i_total)

    with open(f'{path}_caption.npy', 'wb') as f:
      np.save(f, z_t_total)
      

def load_features():
   feats = []
   for i in FEATURE_PATH:
      with open(i, 'rb') as f:
         feats.append(np.load(f, allow_pickle=True))
         
   return feats
        
  