import os
import pickle as pk
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%d-%m-%Y %H:%M:%S')

import numpy as np

import torch
from torch.utils.data import Dataset

from map_patch import MapPatch

class GeoPatchDataset(Dataset):
    def __init__(self, query, key, context, empty_contexts=None):
        self.query = query
        self.key = key
        self.context = context
        self.empty_contexts = empty_contexts

    def __len__(self):
        return len(self.query)

    def __getitem__(self, i):

        if isinstance(i, slice):
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self.query)
            step = i.step if i.step else 1

            return [(self.query[j].patch, (self.key[j].patch, self.context[j].patch)) for j in range(start, stop, step)]

        return self.query[i].patch, np.stack([self.key[i].patch, self.context[i].patch], axis=0)

    def get(self, i):
        if isinstance(i, slice):
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self.query)
            step = i.step if i.step else 1

            return [(self.query[j], (self.key[j], self.context[j])) for j in range(start, stop, step)]

        return self.query[i], self.key[i], self.context[i]

    @staticmethod
    def get_surrounding_indices(index, shift, max_index):
        i, j = index
        indices = [(x, y)
                   for x in range(i - shift, i + shift + 1)
                   for y in range(j - shift, j + shift + 1)
                   if 0 <= x <= max_index[0]
                   and 0 <= y <= max_index[1]
                   and (x, y) != index]

        return indices

    @staticmethod
    def get_geographical_patch_dict(patch_datasets):
        geographical_patch_dict = {}
        max_index = [0, 0]

        for patch_dataset in patch_datasets:
            for patch in patch_dataset.X_1:
                patch_name = os.path.basename(patch.origin_map)
                patch_dict = geographical_patch_dict.get(patch_name, {})
                patch_index = patch.patch_index

                if patch_index not in patch_dict:
                    patch_dict[patch_index] = patch

                geographical_patch_dict[patch_name] = patch_dict

                if patch_index[0] > max_index[0]:
                    max_index[0] = patch_index[0]

                if patch_index[1] > max_index[1]:
                    max_index[1] = patch_index[1]

            for patch in patch_dataset.X_2:
                patch_name = os.path.basename(patch.origin_map)
                patch_dict = geographical_patch_dict.get(patch_name, {})
                patch_index = patch.patch_index

                if patch_index not in patch_dict:
                    patch_dict[patch_index] = patch

                geographical_patch_dict[patch_name] = patch_dict

                if patch_index[0] > max_index[0]:
                    max_index[0] = patch_index[0]

                if patch_index[1] > max_index[1]:
                    max_index[1] = patch_index[1]

        return geographical_patch_dict, max_index

    @classmethod
    def from_dataset(cls, patch_dataset, shift, n_contexts, geographical_patch_dict, max_index):

        np.random.seed(23)

        query = []
        key = []
        context = []
        empty_contexts = 0

        for i in range(len(patch_dataset)):
            query_patch = patch_dataset.X_1[i]
            patch_name = os.path.basename(query_patch.origin_map)

            key_patch = patch_dataset.X_2[i]

            # get indices corresponding to patches around the central patch
            surrounding_indices = set(GeoPatchDataset.get_surrounding_indices(query_patch.patch_index,
                                                                              shift=shift,
                                                                              max_index=max_index))

            # cap the number of possible contexts to number of surrounding patches
            # for example, if we ask for n_contexts = 39274932 and there are only 8
            # surrounding patches, we'll limit the number of contexts added to this
            n_contexts = n_contexts if n_contexts <= len(surrounding_indices) else len(surrounding_indices)

            # get the indices of the actual patches which surround the central patch
            map_indices = set(geographical_patch_dict[patch_name].keys())

            # the valid indices are all those indices which are within a given shift of
            # the central patch, and which actually surround the central patch in the map
            valid_indices = list(surrounding_indices.intersection(map_indices))

            # sample from the valid indices
            n_samples = n_contexts if n_contexts <= len(valid_indices) else len(valid_indices)
            context_indices = [valid_indices[i] for i in np.random.choice(len(valid_indices), n_samples, replace=False)]

            for j in range(n_contexts):
                if j < len(context_indices):
                    context_index = context_indices[j]
                    context_patch = geographical_patch_dict[patch_name][context_index]

                    query.append(query_patch)
                    key.append(key_patch)
                    context.append(context_patch)
                else:
                    context_patch = MapPatch(patch=np.zeros_like(query_patch.patch),
                                             patch_index=(-1, -1),
                                             origin_map=query_patch.origin_map)

                    query.append(query_patch)
                    key.append(key_patch)
                    context.append(context_patch)

                    empty_contexts += 1

        return cls(query, key, context, empty_contexts)

    def save(self, file_name):
        with open(file_name, "wb") as f:
            if file_name.endswith(".pt"):
                torch.save(self, f)
            else:
                pk.dump(self, f)