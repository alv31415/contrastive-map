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

from map_patch import MapPatch


class CanonicalDataset(Dataset):
    """
    Dataset to generate canonical representations of historical maps.
    """

    def __init__(self, historical_patches, canonical_patches, canonical_patch_dict, canonical_key):
        self.historical_patches = historical_patches
        self.canonical_patches = canonical_patches
        self.canonical_patch_dict = canonical_patch_dict
        self.canonical_key = canonical_key

    def __len__(self):
        return len(self.historical_patches)

    def __getitem__(self, i):

        if isinstance(i, slice):
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self.historical_patches)
            step = i.step if i.step else 1

            return [(self.historical_patches[j].patch, self.canonical_patches[j].patch) for j in
                    range(start, stop, step)]

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
            if img.endswith(".png"):
                dot_index = img.index(".")
                folder_number = img[:dot_index]
                if folder_number.isdigit():
                    canonical_patch_list = MapPatch.get_map_patch_list(file_name=os.path.join(canonical_maps_dir, img),
                                                                       patch_width=patch_width,
                                                                       verbose=False)
                    canonical_patch_dict[int(folder_number)] = canonical_patch_list

        return canonical_patch_dict

    @staticmethod
    def get_canonical_patch_names(data_dir, canonical_idx):
        canonical_patch_names = []
        for folder in os.listdir(data_dir):
            if folder.isdigit():
                files = os.listdir(os.path.join(data_dir, folder))
                files = sorted([file for file in files if file.endswith(".png")])

                if len(files) >= 3:
                    canonical_patch_name = files[canonical_idx]
                    canonical_patch_names.append(canonical_patch_name)

        return canonical_patch_names

    @staticmethod
    def get_canonical_patch_dict_from_dataset(patch_datasets, canonical_idx, data_dir=None):
        canonical_patch_dict = {}
        if data_dir is None:
            data_dir = os.path.abspath(os.path.join(patch_datasets[0].X_1[0].origin_map, os.pardir, os.pardir))

        logging.info(f"Fetching canonical patch names from {data_dir}")
        canonical_patch_names = CanonicalDataset.get_canonical_patch_names(data_dir=data_dir,
                                                                           canonical_idx=canonical_idx)

        for patch_dataset in patch_datasets:
            for patch in patch_dataset.X_1:
                patch_name = os.path.basename(patch.origin_map)
                patch_folder = CanonicalDataset.get_folder(patch)

                if patch_name in canonical_patch_names:
                    folder_dict = canonical_patch_dict.get(patch_folder, {})

                    if patch.patch_index not in folder_dict:
                        folder_dict[patch.patch_index] = patch
                        canonical_patch_dict[patch_folder] = folder_dict

            for patch in patch_dataset.X_2:
                patch_name = os.path.basename(patch.origin_map)
                patch_folder = CanonicalDataset.get_folder(patch)

                if patch_name in canonical_patch_names:
                    folder_dict = canonical_patch_dict.get(patch_folder, {})

                    if patch.patch_index not in folder_dict:
                        folder_dict[patch.patch_index] = patch
                        canonical_patch_dict[patch_folder] = folder_dict

        return canonical_patch_dict

    @classmethod
    def from_osm_dir(cls, patch_dataset_dir, canonical_maps_dir):
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

        iter_range = len(patch_dataset.X_1)

        logging.info(f"Matching historical patches with canonical patches ({iter_range} iterations)")

        seen_dict = {}
        for i in range(iter_range):
            historical_patch_1 = patch_dataset.X_1[i]
            historical_patch_2 = patch_dataset.X_2[i]

            patch_index = historical_patch_1.patch_index

            keep_patch_1 = False
            seen_patch_1 = seen_dict.get(historical_patch_1.origin_map, [])
            if historical_patch_1.patch_index not in seen_patch_1:
                seen_patch_1.append(historical_patch_1.patch_index)
                seen_dict[historical_patch_1.origin_map] = seen_patch_1
                keep_patch_1 = True

            keep_patch_2 = False
            seen_patch_2 = seen_dict.get(historical_patch_2.origin_map, [])
            if historical_patch_2.patch_index not in seen_patch_2:
                seen_patch_2.append(historical_patch_2.patch_index)
                seen_dict[historical_patch_2.origin_map] = seen_patch_2
                keep_patch_2 = True

            canonical_map_name = CanonicalDataset.get_folder(historical_patch_1)
            canonical_map_patches = canonical_patch_dict[canonical_map_name]

            canonical_patch = canonical_map_patches[patch_index[0] * n_cols + patch_index[1]]

            assert canonical_patch.patch_index == patch_index

            if keep_patch_1:
                historical_patches.append(historical_patch_1)
                canonical_patches.append(canonical_patch)
            if keep_patch_2:
                historical_patches.append(historical_patch_2)
                canonical_patches.append(canonical_patch)

        return cls(historical_patches, canonical_patches, canonical_patch_dict, canonical_key="osm")

    @classmethod
    def from_os_dataset(cls, patch_dataset, canonical_idx, data_dir=None, canonical_patch_dict=None,
                        remove_copies=False):

        c = 0

        historical_patches = []
        canonical_patches = []
        seen_dict = {}

        if canonical_patch_dict is None:
            logging.info(
                f"Creating dictionary for canonical patches, based on map number {canonical_idx + 1} in {data_dir}.")
            canonical_patch_dict = CanonicalDataset.get_canonical_patch_dict_from_dataset(
                patch_datasets=[patch_dataset],
                canonical_idx=canonical_idx,
                data_dir=data_dir)

        iter_range = len(patch_dataset.X_1)
        logging.info(f"Matching historical patches with canonical patches ({iter_range} iterations)")
        for i in range(iter_range):
            historical_patch_1 = patch_dataset.X_1[i]
            historical_patch_2 = patch_dataset.X_2[i]

            keep_patch_1 = False
            seen_patch_1 = seen_dict.get(historical_patch_1.origin_map, [])
            if historical_patch_1.patch_index not in seen_patch_1:
                seen_patch_1.append(historical_patch_1.patch_index)
                seen_dict[historical_patch_1.origin_map] = seen_patch_1
                keep_patch_1 = True

            keep_patch_2 = False
            seen_patch_2 = seen_dict.get(historical_patch_2.origin_map, [])
            if historical_patch_2.patch_index not in seen_patch_2:
                seen_patch_2.append(historical_patch_2.patch_index)
                seen_dict[historical_patch_2.origin_map] = seen_patch_2
                keep_patch_2 = True

            assert historical_patch_1.patch_index == historical_patch_2.patch_index

            patch_index = historical_patch_1.patch_index

            canonical_map_folder = CanonicalDataset.get_folder(historical_patch_1)

            if canonical_map_folder in canonical_patch_dict:
                if patch_index in canonical_patch_dict[canonical_map_folder]:
                    canonical_patch = canonical_patch_dict[canonical_map_folder][patch_index]

                    if remove_copies:
                        if historical_patch_1.origin_map != canonical_patch.origin_map and keep_patch_1:
                            historical_patches.append(historical_patch_1)
                            canonical_patches.append(canonical_patch)
                        if historical_patch_2.origin_map != canonical_patch.origin_map and keep_patch_2:
                            historical_patches.append(historical_patch_2)
                            canonical_patches.append(canonical_patch)
                    else:
                        if keep_patch_1:
                            historical_patches.append(historical_patch_1)
                            canonical_patches.append(canonical_patch)
                        if keep_patch_2:
                            historical_patches.append(historical_patch_2)
                            canonical_patches.append(canonical_patch)
                else:
                    c += 1

        print(f"{c} matches not found")

        return cls(historical_patches, canonical_patches, canonical_patch_dict, canonical_key="os")

    def save(self, file_name):
        with open(file_name, "wb") as f:
            if file_name.endswith(".pt"):
                torch.save(self, f)
            else:
                pk.dump(self, f)
