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
import pkgutil
import importlib.resources
from gallirallus.dataops import RESOURCE_path

# resource_p = importlib.resources.Resource('gallirallus.resource')
print(RESOURCE_path)

# resource_path = pkgutil.get_data(__name__, "templates/temp_file")


class NSD_ann():
    """Annotation class for linking NSD images to COCO dataset
    TODO: merge coco train and val data during loading
    to simplify correspondance
    """
    def __init__(self, platform: str, local_annot_path=None) -> None:
        """Initialize and load files.

        Args:
            platform (str): Either colab or assumed you work locally.
        """
        self.platform = platform
        self.annot_path = local_annot_path

        # in colab, install modules and mount gdrive
        if platform == 'colab':
            eval('!pip install pycocotools')
            from pycocotools.coco import COCO  # noqa
            from google.colab import drive  # noqa
            drive.mount('/content/drive/', force_remount=True)
            self.annot_path = '/content/drive/MyDrive/coco_2017_annotations'

        self.load_ann()
        self.load_nsd_info()
        self.nsd_coco_hash_table()
        print('loading success!')

    def _is_array_like(self, obj):
        return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

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
        self.coco_test = COCO(Path(self.annot_path).joinpath('instances_val2017.json'))


    def load_nsd_info(self):
        """Load csv with experimental info on the NSD dataset into a pandas DataFrame
        self.nsd_info. Contains 40 columns of info about each image used, ids, repetitions for
        each subject etc.

        See here for description of the column values:
        https://cvnlab.slite.page/p/NKalgWd__F/Experiments#bf18f984
        """
        print('loading NSD image infos')
        load_path = self.annot_path if self.platform=='colab' else RESOURCE_path
        self.nsd_info = pd.read_csv(Path(load_path).joinpath('nsd_stim_info_merged.csv'))

    def nsd_coco_hash_table(self) -> dict:
        """Creates hash table as a dictionary where the key is the nsd ID
        and values are tuple of (coco ID, coco_split dataset)

        Returns:
            dict: nsd ID: (coco ID, split) for all 73 000 NSD images
        """
        print('Creating hash table between NSD img id and cocoId and coco split')
        self.hash = {}
        for i in range(1, len(self.nsd_info)):
            self.hash[i] = (self.nsd_info.loc[i]['cocoId'],
                            self.nsd_info.loc[i]['cocoSplit'],
                            )
        return self.hash

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
            coco_id, split = self.hash[id]
            if split == 'train2017':
                cats = [k['category_id'] for k in self.coco_train.imgToAnns[coco_id]]
            else:
                cats = [k['category_id'] for k in self.coco_test.imgToAnns[coco_id]]
            
            self.cats[id] = cats
        return self.cats
    

    ###### depraceted ######
    ########################
    def coco_id_from_nsd(self, ids: list[int]) -> tuple:
        ids = ids if self._is_array_like(ids) else [ids]

        self.ids_coco = tuple(self.nsd_info.loc[ids]['cocoId'])
        self.splits = tuple(self.nsd_info.loc[ids]['cocoSplit'])
        return self.ids_coco, self.splits



if __name__ == '__main__':
    # Need to be changed depending on where is your coco annotation file
    ann = NSD_ann(platform='local',
                  local_annot_path=Path('C:/Users/David Palecek/Documents/UAlg/Neuromatch/project/coco_ann_trainval2017/annotations'),
                  )
    # print('######## keys of COCO train ##########')
    # print(ann.coco_train.__dict__.keys())
    # print('######## keys of NSD info ##########')
    # print(ann.nsd_info.info())

    # get category IDs for first three NSD images.
    categories = ann.nsd_cat([1, 2, 3])
    print(categories)


