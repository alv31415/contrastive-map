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
    def __init__(self, X_1, X_2):
        self.X_1 = X_1
        self.X_2 = X_2
    
    def __len__(self):
        return len(self.X_1)
    
    def __getitem__(self, i):
        
        if isinstance(i, slice):
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self.patches)
            step = i.step if i.step else 1
            
            return [(self.X_1[j], self.X_2[j]) for j in range(start, stop, step)]
        
        return (self.X_1[i].patch, self.X_2[i].patch)
    
    @classmethod
    def from_dir(cls, map_directory, patch_width, verbose = False):
        X_1 = []
        X_2 = []

        logging.info(f"Directories found: {os.listdir(map_directory)}")

        for folder in os.listdir(map_directory):
            if folder.isdigit():
                if verbose:
                    logging.info(f"Fetching patches from folder: {folder}")
                    
                directory = os.path.join(map_directory, folder)
                patch_list = CLPatchDataset.get_patch_list_from_dir(directory, patch_width = patch_width, verbose = False)
                x_1, x_2 = CLPatchDataset.get_matching_patch_list(patch_list)
                X_1.extend(x_1)
                X_2.extend(x_2)
            
        return cls(X_1, X_2)
    
    @staticmethod
    def index_sampler(indices, n_samples = 4):
        return [indices[i] for i in np.random.choice(len(indices), n_samples, replace=False)]

    @staticmethod
    def get_patch_list_from_dir(directory, patch_width, verbose = False):
        patches = []

        for file in os.listdir(directory):
            if file.endswith("tif"):
                file_name = f"{directory}/{file}"
                patches.append(MapPatch.get_map_patch_list(file_name = file_name, 
                                                           patch_width = patch_width, 
                                                           verbose = verbose))

        return patches
    
    @staticmethod
    def is_empty_patch(patch):
        gray = cv.cvtColor(np.array(patch), cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray, (3,3), 0)
        edges = cv.Canny(gray, 50, 150)
        black_pixels = np.where(edges == 255)

        return len(black_pixels[0]) <= 20
    
    @staticmethod
    def get_matching_patch_list(patch_list):
        n_samples = len(patch_list)
        indices = [(i,j) for i in range(n_samples) for j in range(i) if i != j]

        x_1 = []
        x_2 = []

        for i in range(len(patch_list[0])):
            #sample_indices = CLPatchDataset.index_sampler(indices, n_samples)
            sample_indices = [(i,i+1) for i in range(n_samples-1)]
            for index in sample_indices:
                try:
                    patch_1 = patch_list[index[0]][i]
                    patch_2 = patch_list[index[1]][i]
                    if not CLPatchDataset.is_empty_patch(patch_1.patch) and not CLPatchDataset.is_empty_patch(patch_2.patch):
                        x_1.append(patch_1)
                        x_2.append(patch_2)
                except IndexError:
                    print(f"Faulty generated index: {index}")
                    print(f"Index i: {i}")
                    print(f"len(patch_list): {len(patch_list)}")
                    print(f"len(patch_list[index[0]]): {len(patch_list[index[0]])}")
                    print(f"len(patch_list[index[1]]): {len(patch_list[index[1]])}")
                    raise ValueError()

        return x_1, x_2
    
    def save(self, file_name):
        with open(f"{file_name}.pk", "wb") as f:
            pk.dump(self, f)