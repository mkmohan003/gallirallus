from torchvision import models
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import clip
import tqdm
import lzma, pickle

# from configs import *
from dataset.algonuts_dataset import AlgonutsDataset, ArgObj


MAIN_PATH = 'C:/Users/David Palecek/Documents/UAlg/Neuromatch/project/algonauts_2023_tutorial_data/'#'/content/drive/MyDrive/algonauts_2023_tutorial_data/'
ANNOTATION_PATH = 'C:/Users/David Palecek/Documents/UAlg/Neuromatch/project/coco_ann_trainval2017/annotations/' #'/content/drive/MyDrive/annotations'
NSD_STIM_INFO_PATH = 'C:/Users/David Palecek/Documents/UAlg/Neuromatch/project/github/nsd/gallirallus/resource/' #'/content/'

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
    model_name = CLIP_AVAL_MODELS[0]

    # load model
    print('Loading model...')
    model, preprocess = clip.load(model_name, device=device)

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

      features = []
      for img, _ in iter(algonuts_train_loader):
         features.append(model.encode_image(img))
      
      all_features = np.array(all_features)
      print('All features shape', all_features.shape)
      ## I think in case of shuffle data as in algonauts tutorial, saving features at this point is wrong.
      # Needs to be done after PCA
      with lzma.open(f'visual_features_{model_name}.npz', 'wb') as f:
          pickle.dump(all_features, f)
         

      z_t = model.encode_text(tokens)
      z_i = model.encode_image(img) 

      print(z_t.size())
      print(z_i.size())


if __name__ == "__main__":
    main()