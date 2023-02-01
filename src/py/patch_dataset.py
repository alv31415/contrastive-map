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

from torch.utils.data import Dataset

from map_patch import MapPatch

class CLPatchDataset(Dataset):
    def __init__(self, X_1, X_2, removed_patches = None):
        self.X_1 = X_1
        self.X_2 = X_2
        self.removed_patches = []

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

        logging.info(f"Directories found: {sorted(os.listdir(map_directory))}")
        logging.info(
            f"Files in first directory: {os.listdir(os.path.join(map_directory, os.listdir(map_directory)[0]))}")


        for folder in os.listdir(map_directory):
            if folder.isdigit():
                if verbose:
                    logging.info(f"Fetching patches from folder: {folder}")

                directory = os.path.join(map_directory, folder)
                patch_list = CLPatchDataset.get_patch_list_from_dir(directory, patch_width=patch_width, verbose=False)
                x_1, x_2, _removed_patches = CLPatchDataset.get_matching_patch_list(patch_list)
                X_1.extend(x_1)
                X_2.extend(x_2)
                removed_patches.extend(_removed_patches)


        logging.info(f"Generated {len(X_1)} positive pairs, after removing {len(removed_patches)} positive pairs.")

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
    def is_empty_patch(patch):
        gray = cv.cvtColor(np.array(patch), cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        edges = cv.Canny(gray, 50, 150)

        black_pixels = np.where(edges == 255)

        if len(black_pixels[0]) <= 20:
            return True
        else:
            for row in range(10):
                if np.all(edges[row, :] == 255):
                    return True

        return False

    @staticmethod
    def get_matching_patch_list(patch_list):
        n_samples = len(patch_list)

        x_1 = []
        x_2 = []

        sample_indices = [(k, j) for k in range(n_samples) for j in range(k) if k != j]
        n_patches = set([len(patch_list[i]) for i in range(n_samples)])
        removed_patches = []

        if len(n_patches) != 1:
            logging.error(f"Found different patch list lengths: {n_patches}. Patches won't be added to the dataset.")
            return x_1, x_2, len(removed_patches)

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
                        if x_1 not in removed_patches:
                            removed_patches.append(x_1)

                        if x_2 not in removed_patches:
                            removed_patches.append(x_2)

                except IndexError:
                    logging.error(
                        f"Error processing patches. Tried fetching patches at index {i}, but patch lists have lengths {len(patch_list[index[0]])} and {len(patch_list[index[1]])}")
                    raise ValueError()

        return x_1, x_2, removed_patches

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pk.dump(self, f)