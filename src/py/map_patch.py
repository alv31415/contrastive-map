import PIL
from PIL import Image
import logging

import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify,unpatchify

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.basicConfig(format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%d-%m-%Y %H:%M:%S')

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

Image.MAX_IMAGE_PIXELS = 933120000

class MapPatch():
    def __init__(self, patch, patch_index, origin_map):
        self.patch = patch
        self.patch_index = patch_index
        self.origin_map = origin_map
        self.patch_shift = None
        
    @staticmethod
    def get_map_patches(file_name, patch_width, verbose = True):
        tif_map = Image.open(file_name)
        tif_map_np = np.array(tif_map)
        
        tif_map_patches = patchify(image = tif_map_np, 
                                   patch_size = (patch_width, patch_width, 3),
                                   step = patch_width)

        if verbose:
            logging.info(f"{np.prod(tif_map_patches.shape[:2]):,} patches from {file_name} generated with shape {tif_map_patches.shape}")

        return tif_map_np, tif_map_patches
    
    @staticmethod
    def get_map_patch_list(file_name, patch_width, verbose = True):
        _, tif_map_patches = MapPatch.get_map_patches(file_name, 
                                                      patch_width, 
                                                      verbose = verbose)
        
        patches = [MapPatch(tif_map_patches[i,j,0], patch_index = (i,j), origin_map = file_name)
                  for i in range(tif_map_patches.shape[0])
                  for j in range(tif_map_patches.shape[1])]
                
        return patches
    
    def get_map_px(self):
        """
        Gets location of the top left pixel of the patch in the original image.
        """
        patch_size = self.patch.shape[0]
        row = self.patch_index[0]
        col = self.patch_index[1]
        
        return (row * patch_size, col * patch_size)
    
    def show(self, verbose = True):
        """
        Shows the patch.
        """
        fig, ax = plt.subplots()
        ax.imshow(self.patch)
        
        if verbose:
            ax.set_title(f"Patch at {self.patch_index} from {self.origin_map}.")
            
        plt.show()