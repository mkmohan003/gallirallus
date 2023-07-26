#!/usr/bin/env python3

'''
NSD_ann class matches annotations of the COCO 2017 dataset to NSD
indexing.

2 files needed:
    * NSD images info file (csv, 11MB): 'nsd_stim_info_merged.csv'
    * COCO annotation files (json, 448MB train, 19MB val)

The csv is included in this repo in resource folder
Path to COO annotations need to be specified in the case of local runtime

In case of running colab environment, add this gdrive by "Add a shortcut to Drive",
otherwise it is not going to work for sure :)
https://drive.google.com/drive/folders/1AJl_4oi0b1jevnS1jY6jnBckPGUHaXzX?usp=sharing

Both NSD and COCO files will be loaded from there.
'''

__author__ = 'David Palecek (david@stanka.de)'

import numpy as np
import pandas as pd
from pathlib import Path
from pycocotools.coco import COCO


class NSD_ann():
    def __init__(self, annotations_path: str, nsd_stim_info_path: str, load_caption: bool) -> None:
        self.annot_path = annotations_path
        self.load_path = nsd_stim_info_path

        if load_caption:
            self.load_captions()

        self.load_ann()
        self.load_nsd_info()
        self.nsd_coco_hash_table()
        print('loading success!')

    def _is_array_like(self, obj):
        return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

    def load_captions(self):   
        # loading
        self.coco_caption_train = COCO(Path(self.annot_path).joinpath('captions_train2017.json'))
        self.coco_caption_test = COCO(Path(self.annot_path).joinpath('captions_val2017.json'))
       
    def load_ann(self):
        """Loads COCO annotations into self.coco_train and
        self.coco_test.

        The link below shows how to use pycoco methods to retrieve categories
        or, captions, segmentations etc.
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
        """
        print('loading COCO image annotations')

        # loading
        self.coco_train = COCO(Path(self.annot_path).joinpath('instances_train2017.json'))
        self.coco_test = COCO(Path(self.annot_path).joinpath('captions_val2017.json'))

    def load_nsd_info(self):
        """Load csv with experimental info on the NSD dataset into a pandas DataFrame
        self.nsd_info. Contains 40 columns of info about each image used, ids, repetitions for
        each subject etc.

        See here for description of the column values:
        https://cvnlab.slite.page/p/NKalgWd__F/Experiments#bf18f984
        """
        print('loading NSD image infos')
        self.nsd_info = pd.read_csv(Path(self.load_path).joinpath('nsd_stim_info_merged.csv'))

    def nsd_coco_hash_table(self) -> dict:
        """Creates hash table as a dictionary where the key is the nsd ID
        and values are tuple of (coco ID, coco_split dataset)

        Returns:
            dict: nsd ID: (coco ID, split) for all 73 000 NSD images
        """
        print('Creating hash table between NSD img id and cocoId and coco split')
        self.hash_table = {}
        for i in range(1, len(self.nsd_info)):
            self.hash_table[i] = (self.nsd_info.loc[i]['cocoId'],
                            self.nsd_info.loc[i]['cocoSplit'],
                            )
        return self.hash_table

    def nsd_cat(self, ids: list[int]) -> dict:
        """Retrieve Cat numbers for give list of NSD images identified
        by their IDs

        Args:
            ids (list[int]): list of NSD ids

        Returns:
            dict: NSD id: list of cat NUmbers
        """
        # nsd ids
        ids = ids if self._is_array_like(ids) else [ids]

        self.cats = {}
        for id in ids:
            coco_id, split = self.hash_table[id]
            if split == 'train2017':
                cats = [k['category_id'] for k in self.coco_train.imgToAnns[coco_id]]
            else:
                cats = [k['category_id'] for k in self.coco_test.imgToAnns[coco_id]]
            
            self.cats[id] = cats
        return self.cats
    
    def nsd_captions(self, id):
        coco_id, split = self.hash_table[id]
        if split == 'train2017':
            captions = [k['caption'] for k in self.coco_caption_train.imgToAnns[coco_id]]
        else:
            captions = [k['caption'] for k in self.coco_caption_test.imgToAnns[coco_id]]
        
        return captions
















