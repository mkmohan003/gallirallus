from torchvision import models
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import clip

from dataset.algonuts_dataset import AlgonutsDataset, ArgObj


MAIN_PATH = '/content/drive/MyDrive/algonauts_2023_tutorial_data/'
ANNOTATION_PATH = '/content/annotations'
NSD_STIM_INFO_PATH = '/content/'
CLIP_AVAL_MODELS = ['RN50',
                    'RN101',
                    'RN50x4',
                    'RN50x16',
                    'RN50x64',
                    'ViT-B/32',
                    'ViT-B/16',
                    'ViT-L/14',
                    'ViT-L/14@336px']


def main():
    # setup
    print('Setup...')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    print('Loading model...')
    model, preprocess = clip.load(CLIP_AVAL_MODELS[0], device=device)

    # load dataset
    args = ArgObj(MAIN_PATH, '01')
    algonuts_train_set = AlgonutsDataset(args.data_dir, 'train', load_annotations=True, device=device, 
                                          clip_preprocess=preprocess, annotations_path=ANNOTATION_PATH, 
                                          nsd_stim_path=NSD_STIM_INFO_PATH)
    algonuts_train_loader = DataLoader(algonuts_train_set, batch_size=1)

    img, lh_fmri, rh_fmri, tokens = next(iter(algonuts_train_loader))
    # print(tokens)

    # Extract features
    with torch.no_grad():
      z_t = model.encode_text(tokens)
      z_i = model.encode_image(img) 

      print(z_t.size())
      print(z_i.size())





if __name__ == "__main__":
    main()