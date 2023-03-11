import os
import logging
import pickle as pk

import numpy as np
import cv2 as cv

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%d-%m-%Y %H:%M:%S')

import torch
from torch.utils.data import Dataset, random_split

from map_patch import MapPatch


class CLPatchDataset(Dataset):
    def __init__(self, X_1, X_2, removed_patches=None):
        self.X_1 = X_1
        self.X_2 = X_2
        self.removed_patches = removed_patches

    def __len__(self):
        return len(self.X_1)

    def __getitem__(self, i):

        if isinstance(i, slice):
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self.X_1)
            step = i.step if i.step else 1

            return [(self.X_1[j], self.X_2[j]) for j in range(start, stop, step)]

        return (self.X_1[i].patch, self.X_2[i].patch)

    def get_removed_patch(self, i):
        return self.removed_patches[i]

    @classmethod
    def from_dir(cls, map_directory, patch_width, verbose=False):
        X_1 = []
        X_2 = []
        removed_patches = []
        removed_ppairs = 0

        logging.info(f"Directories found: {sorted(os.listdir(map_directory))}")
        logging.info(
            f"Files in first directory: {os.listdir(os.path.join(map_directory, os.listdir(map_directory)[0]))}")

        for folder in os.listdir(map_directory):
            if folder.isdigit():
                if verbose:
                    logging.info(f"Fetching patches from folder: {folder}")

                directory = os.path.join(map_directory, folder)
                patch_list = CLPatchDataset.get_patch_list_from_dir(directory, patch_width=patch_width, verbose=False)
                x_1, x_2, _removed_patches, _removed_ppairs = CLPatchDataset.get_matching_patch_list(patch_list)
                X_1.extend(x_1)
                X_2.extend(x_2)
                removed_patches.extend(_removed_patches)
                removed_ppairs += _removed_ppairs

        logging.info(f"Generated {len(X_1)} positive pairs, after removing {removed_ppairs} positive pairs.")

        return cls(X_1, X_2, removed_patches)

    @staticmethod
    def index_sampler(indices, n_samples=4):
        return [indices[i] for i in np.random.choice(len(indices), n_samples, replace=False)]

    @staticmethod
    def get_patch_list_from_dir(directory, patch_width, verbose=False):
        patches = []

        for file in os.listdir(directory):
            if file.endswith("png") or file.endswith("tif"):
                file_name = os.path.join(directory, file)
                patches.append(MapPatch.get_map_patch_list(file_name=file_name,
                                                           patch_width=patch_width,
                                                           verbose=verbose))

        return patches

    @staticmethod
    def count_black_pixels(edges):
        black_pixel_idx = np.where(edges == 255)
        return len(black_pixel_idx[0])

    @staticmethod
    def get_edges(patch):
        gray = cv.cvtColor(np.array(patch), cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        return cv.Canny(gray, 50, 150)

    @staticmethod
    def is_empty_patch(patch):
        edges = CLPatchDataset.get_edges(patch)
        patch_width = patch.shape[0]

        n_black_pixels = CLPatchDataset.count_black_pixels(edges)

        min_black_pixels = int(0.01 * patch_width * patch_width)
        edge_pixel_range = int(0.1 * patch_width)
        max_edge_pixels = int(0.5 * n_black_pixels)

        if n_black_pixels <= min_black_pixels:
            return True
        else:
            for row in range(edge_pixel_range):
                if CLPatchDataset.count_black_pixels(edges[row, :]) >= max_edge_pixels \
                        or CLPatchDataset.count_black_pixels(edges[len(edges) - 1 - row, :]) >= max_edge_pixels \
                        or CLPatchDataset.count_black_pixels(edges[:, row]) >= max_edge_pixels \
                        or CLPatchDataset.count_black_pixels(edges[:, len(edges) - 1 - row]) >= max_edge_pixels:
                    return True

        return False

    @staticmethod
    def get_matching_patch_list(patch_list):
        n_samples = len(patch_list)

        x_1 = []
        x_2 = []
        removed_patches = []
        removed_ppairs = 0

        sample_indices = [(k, j) for k in range(n_samples) for j in range(k) if k != j]
        n_patches = set([len(patch_list[i]) for i in range(n_samples)])

        if len(n_patches) != 1:
            logging.error(f"Found different patch list lengths: {n_patches}. Patches won't be added to the dataset.")
            return x_1, x_2, removed_patches

        for i in range(len(patch_list[0])):
            for index in sample_indices:
                try:
                    patch_1 = patch_list[index[0]][i]
                    patch_2 = patch_list[index[1]][i]
                    if not CLPatchDataset.is_empty_patch(patch_1.patch) and not CLPatchDataset.is_empty_patch(
                            patch_2.patch):
                        x_1.append(patch_1)
                        x_2.append(patch_2)
                    else:

                        removed_ppairs += 1

                        if patch_1 not in removed_patches:
                            removed_patches.append(patch_1)

                        if patch_2 not in removed_patches:
                            removed_patches.append(patch_2)

                except IndexError:
                    logging.error(
                        f"Error processing patches. Tried fetching patches at index {i}, but patch lists have lengths {len(patch_list[index[0]])} and {len(patch_list[index[1]])}")
                    raise ValueError()

        return x_1, x_2, removed_patches, removed_ppairs

    def get_patch_dict(self):
        patch_dict = {}

        for i in range(len(self.X_1)):
            patch = self.X_1[i]

            assert patch.patch_index == self.X_2[i].patch_index

            patch_folder = os.path.dirname(patch.origin_map)

            assert patch_folder == os.path.dirname(self.X_2[i].origin_map)

            if patch_folder in patch_dict:
                patch_dict[patch_folder].add(patch.patch_index)
            else:
                patch_dict[patch_folder] = {patch.patch_index}

        return patch_dict

    def get_split_dict(self, p_train, p_validation, seed=23):
        np.random.seed(seed)

        patch_dict = self.get_patch_dict()
        split_dict = {patch_folder: {"train": [], "validation": [], "test": []} for patch_folder in patch_dict.keys()}

        assert p_train + p_validation <= 1

        p_test = 1 - p_train - p_validation

        split_proportions = [p_train, p_validation, p_test]

        for patch_folder, patch_indices in patch_dict.items():
            assert len(split_dict[patch_folder]["train"]) == 0
            assert len(split_dict[patch_folder]["validation"]) == 0
            assert len(split_dict[patch_folder]["test"]) == 0

            patch_indices = list(patch_indices)

            n_samples = len(patch_indices)
            split_sizes = np.multiply(split_proportions, n_samples).astype(int)
            split_sizes[0] += n_samples - np.sum(split_sizes)

            train_indices = np.random.choice(n_samples, size=split_sizes[0], replace=False)
            remaining_indices = np.setdiff1d(np.arange(n_samples), train_indices)
            validation_indices = np.random.choice(remaining_indices, size=split_sizes[1], replace=False)
            test_indices = np.setdiff1d(remaining_indices, validation_indices)

            assert len(train_indices) + len(validation_indices) + len(test_indices) == n_samples
            assert np.intersect1d(train_indices, validation_indices).size == 0 \
                   and np.intersect1d(validation_indices, test_indices).size == 0 \
                   and np.intersect1d(train_indices, test_indices).size == 0

            split_dict[patch_folder]["train"] = [patch_indices[i] for i in train_indices]
            split_dict[patch_folder]["validation"] = [patch_indices[i] for i in validation_indices]
            split_dict[patch_folder]["test"] = [patch_indices[i] for i in test_indices]

        return split_dict

    def unique_split(self, p_train=0.8, p_validation=0.1, p_test=0.1, seed=23):
        split_dict = self.get_split_dict(p_train=p_train,
                                         p_validation=p_validation,
                                         seed=seed)

        train_X_1 = []
        train_X_2 = []

        validation_X_1 = []
        validation_X_2 = []

        test_X_1 = []
        test_X_2 = []

        for i in range(len(self.X_1)):
            patch_1 = self.X_1[i]
            patch_2 = self.X_2[i]

            patch_index = patch_1.patch_index

            assert patch_index == patch_2.patch_index

            patch_folder = os.path.dirname(patch_1.origin_map)

            assert patch_folder == os.path.dirname(patch_2.origin_map)

            if patch_index in split_dict[patch_folder]["train"]:
                train_X_1.append(patch_1)
                train_X_2.append(patch_2)
            elif patch_index in split_dict[patch_folder]["validation"]:
                validation_X_1.append(patch_1)
                validation_X_2.append(patch_2)
            elif patch_index in split_dict[patch_folder]["test"]:
                test_X_1.append(patch_1)
                test_X_2.append(patch_2)
            else:
                raise ValueError(f"Patch index {patch_index} not found in any of the train, validation or test splits")

        train_set = CLPatchDataset(train_X_1, train_X_2, removed_patches=self.removed_patches)
        validation_set = CLPatchDataset(validation_X_1, validation_X_2, removed_patches=self.removed_patches)
        test_set = CLPatchDataset(test_X_1, test_X_2, removed_patches=self.removed_patches)

        return train_set, validation_set, test_set

    def random_split(self, p_train=0.8, p_validation=0.1, seed=23):
        p_test = 1 - p_train - p_validation
        lengths = np.multiply([p_train, p_validation, p_test], len(self.X_1)).astype(int)
        lengths[0] += len(self.X_1) - np.sum(lengths)

        train_data, val_data, test_data = random_split(self,
                                                       lengths=lengths,
                                                       generator=torch.Generator().manual_seed(seed))

        train_X_1 = [self.X_1[i] for i in train_data.indices]
        train_X_2 = [self.X_2[i] for i in train_data.indices]

        val_X_1 = [self.X_1[i] for i in val_data.indices]
        val_X_2 = [self.X_2[i] for i in val_data.indices]

        test_X_1 = [self.X_1[i] for i in test_data.indices]
        test_X_2 = [self.X_2[i] for i in test_data.indices]

        train_set = CLPatchDataset(train_X_1, train_X_2, removed_patches=self.removed_patches)
        validation_set = CLPatchDataset(val_X_1, val_X_2, removed_patches=self.removed_patches)
        test_set = CLPatchDataset(test_X_1, test_X_2, removed_patches=self.removed_patches)

        return train_set, validation_set, test_set

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pk.dump(self, f)