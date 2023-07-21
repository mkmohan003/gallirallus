import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
import os

from .nsd_annos import NSD_ann

class ArgObj:
  def __init__(self, data_dir, subj):
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)


class AlgonutsDataset(Dataset):
    def __init__(self, data_dir: str, split: str, load_annotations: bool, 
                       annotations_path: str = None, nsd_stim_path: str = None):
        self.load_annotations = load_annotations

        if split == 'train':
          self.images_dir = Path(os.path.join(data_dir, 'training_split', 'training_images'))
          self.fmri_dir = Path(os.path.join(data_dir, 'training_split', 'training_fmri'))
        elif split == 'val':
          self.images_dir = Path(os.path.join(data_dir, 'test_split', 'test_images'))
          self.fmri_dir = Path(os.path.join(data_dir, 'test_split', 'test_fmri'))

        self.image_transform = build_transforms()

        print('loading image data...')
        self.image_data = self.load_image_data()

        print('loading fmri data...')
        self.lh_fmri, self.rh_fmri = self.load_fmri_data()

        if load_annotations:
           assert annotations_path != None
           assert nsd_stim_path != None

           print('loading annotations...')
           self.annotations = NSD_ann(annotations_path, nsd_stim_path, load_caption=True)


    def load_image_data(self):
        return list(self.images_dir.glob('*.*'))

    def load_fmri_data(self):
        lh_fmri = np.load(os.path.join(self.fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(self.fmri_dir, 'rh_training_fmri.npy'))

        return lh_fmri, rh_fmri
      

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx: int):
        # Load the image
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert('RGB')
        
        img = self.image_transform(img)
        lh_fmri = torch.from_numpy(self.lh_fmri[idx])
        rh_fmri = torch.from_numpy(self.rh_fmri[idx])

        if self.load_annotations:
           captions = self.annotations.nsd_captions(idx)
           return img, lh_fmri, rh_fmri
        
        else:
          return img, lh_fmri, rh_fmri


def build_transforms():
  images_transform = T.Compose([
    T.ToTensor()
  ])

  return images_transform