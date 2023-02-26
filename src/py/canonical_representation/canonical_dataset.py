import os
import logging
import pickle as pk

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%d-%m-%Y %H:%M:%S')

import torch
from torch.utils.data import Dataset

from src.map_patch import MapPatch

class CanonicalDataset(Dataset):
    """
    Dataset to generate canonical representations of historical maps.
    """
    def __init__(self, historical_patches, canonical_patches):
        self.historical_patches = historical_patches
        self.canonical_patches = canonical_patches

    def __len__(self):
        return len(self.historical_patches)

    def __getitem__(self, i):

        if isinstance(i, slice):
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self.historical_patches)
            step = i.step if i.step else 1

            return [(self.historical_patches[j].patch, self.canonical_patches[j].patch) for j in range(start, stop, step)]

        return self.historical_patches[i].patch, self.canonical_patches[i].patch

    def get(self, i):
        if isinstance(i, slice):
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self.historical_patches)
            step = i.step if i.step else 1

            return [(self.historical_patches[j], self.canonical_patches[j]) for j in range(start, stop, step)]

        return self.historical_patches[i], self.canonical_patches[i]

    @staticmethod
    def get_folder(patch):
        """
        Retrieves the directory from which the patch originated.
        The directories which contain maps are numerical.
        """
        map_dir = patch.origin_map
        dir_path = os.path.dirname(map_dir)

        subdirs = dir_path.split(os.path.sep)

        return int(subdirs[-1])

    @staticmethod
    def get_canonical_patch_dict(canonical_maps_dir, patch_width):
        """"
        Creates a dictionary, based on the canonical maps stored in canonical_maps_dir.
        The key is the canonical map name (numerical, corresponding to directory names of the historical maps;
        for example, the canonical map corresponding to region 57 is named 57.png).
        The value will be a list of MapPatch objects, the patches of the canonical map, which ahve width patch_width..
        """

        canonical_patch_dict = {}

        for img in os.listdir(canonical_maps_dir):
            dot_index = img.index(".")
            folder_number = img[:dot_index]
            if folder_number.isdigit():
                canonical_patch_list = MapPatch.get_map_patch_list(file_name=os.path.join(canonical_maps_dir, img),
                                                             patch_width=patch_width,
                                                             verbose=False)
                canonical_patch_dict[int(folder_number)] = canonical_patch_list

        return canonical_patch_dict

    @classmethod
    def from_dir(cls, patch_dataset_dir, canonical_maps_dir, debug=False):
        """
        Generates a CanonicalDataset, based on the direcotry where a CLPatchDatset is stored,
        and the directory where canonical maps are stored.
        For each patch in the CLPatchDatset, generates a positive pair with the corresponding patch in canonical style.
        """

        logging.info(f"Fetching original patch dataset from {patch_dataset_dir}")
        with open(patch_dataset_dir, "rb") as f:
            patch_dataset = pk.load(f)

        historical_patches = []
        canonical_patches = []

        patch_width = patch_dataset.X_1[0].patch.shape[0]
        logging.info(f"Creating dictionary for canonical patches, with patch width = {patch_width}")
        canonical_patch_dict = CanonicalDataset.get_canonical_patch_dict(canonical_maps_dir, patch_width)

        n_cols = canonical_patch_dict[1][-1].patch_index[1] + 1

        iter_range = 100 if debug else len(patch_dataset.X_1)

        logging.info(f"Matching historical patches with canonical patches ({iter_range} iterations)")
        for i in range(iter_range):
            historical_patch_1 = patch_dataset.X_1[i]
            historical_patch_2 = patch_dataset.X_2[i]

            patch_index = historical_patch_1.patch_index

            canonical_map_name = CanonicalDataset.get_folder(historical_patch_1)
            canonical_map_patches = canonical_patch_dict[canonical_map_name]

            canonical_patch = canonical_map_patches[patch_index[0] * n_cols + patch_index[1]]

            assert canonical_patch.patch_index == patch_index

            historical_patches.extend([historical_patch_1, historical_patch_2])
            canonical_patches.extend([canonical_patch, canonical_patch])

        return cls(historical_patches, canonical_patches)

    def save(self, file_name):
        with open(file_name, "wb") as f:
            if file_name.endswith(".pt"):
                torch.save(self, f)
            else:
                pk.dump(self, f)
