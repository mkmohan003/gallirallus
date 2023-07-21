from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import os

from configs import *
from dataset.algonuts_dataset import AlgonutsDataset, ArgObj


MAIN_PATH = '/content/drive/MyDrive/algonauts_2023_tutorial_data/'
ANNOTATION_PATH = '/content/drive/MyDrive/annotations'
NSD_STIM_INFO_PATH = '/content/'

def load_model(model_name):
    base_model = getattr(models, model_name)(pretrained=True)
    return base_model


def main():
    model_name = 'alexnet'
    base_model = load_model(model_name)

    # load dataset
    args = ArgObj(MAIN_PATH, '01')
    algonuts_train_set = AlgonutsDataset(args.data_dir, 'train', load_annotations=True, 
                                          annotations_path=ANNOTATION_PATH, nsd_stim_path=NSD_STIM_INFO_PATH)
    algonuts_train_loader = DataLoader(algonuts_train_set, batch_size=1)

    img, lh_fmri, rh_fmri = next(iter(algonuts_train_loader))
    print(img, lh_fmri)



if __name__ == "__main__":
    main()