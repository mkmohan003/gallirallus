from torchvision import models
from torch.utils.data import DataLoader
import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import clip
import lzma, pickle
from argparse import ArgumentParser
import joblib


# from configs import *
from dataset.algonuts_dataset import AlgonutsDataset, ArgObj, fmri_data
from utils import extract_store_features, load_features


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

def build_encoding_model(img_features, lh_fmri, rh_fmri):
    reg_lh = LinearRegression().fit(img_features, lh_fmri)
    reg_rh = LinearRegression().fit(img_features, rh_fmri)

    return reg_lh, reg_rh


def main(args):
    # setup
    print('Setup...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = CLIP_AVAL_MODELS[0]

    # load model
    print('Loading model...')
    model, preprocess = clip.load(model_name, device=device)

    # load dataset
    data_args = ArgObj(MAIN_PATH, '01')
    lh_fmri, rh_fmri = fmri_data(data_args.data_dir)
    algonuts_train_set = AlgonutsDataset(data_args.data_dir, 'train', load_annotations=True, device=device, 
                                          clip_preprocess=preprocess, annotations_path=ANNOTATION_PATH, 
                                          nsd_stim_path=NSD_STIM_INFO_PATH)
    algonuts_train_loader = DataLoader(algonuts_train_set, batch_size=1)

    ### img, lh_fmri, rh_fmri, tokens = next(iter(algonuts_train_loader))

    # Extract features
    if args.extract_features:
        print('Extracting features...')
        extract_store_features(model, algonuts_train_loader, 'subj01_train')

    print('Loading features...')
    features = load_features()

    # build model
    print('Building model...')
    lh_reg, rh_reg = build_encoding_model(features[0], lh_fmri, rh_fmri)
    joblib.dump(lh_reg, 'lh_reg.pkl')
    joblib.dump(rh_reg, 'rh_reg.pkl')


if __name__ == "__main__":
    parser = ArgumentParser('CLIP2Brain')
    parser.add_argument('--extract_features', action='store_true')
    args = parser.parse_args()

    main(args)