import os

import logging
import pickle as pk
import argparse
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format="%(asctime)s %(levelname)-4s %(message)s",
                    level=logging.INFO,
                    datefmt="%d-%m-%Y %H:%M:%S")

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.info(f"Running main & importing modules...")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from patch_dataset import CLPatchDataset
from unet import UNet
from byol import MapBYOL
from simclr import MapSIMCLR
from canonical_dataset import CanonicalDataset

# --------------------------------------------------- PARSER ---------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(description="Self-Supervised Map Embeddings")

    # training params
    parser.add_argument("--batch-size", type=int, default=50, metavar="N",
                        help="input batch size for training (default: 50)")
    parser.add_argument("--patch-size", type=int, default=64, metavar="N",
                        help="size of patches for dataset (default: 64)")
    parser.add_argument("--epochs", type=int, default=15, metavar="N",
                        help="number of epochs to train (default: 15)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=23, metavar="S",
                        help="random seed (default: 23)")
    parser.add_argument("--logs-per-epoch", type=int, default=50, metavar="N",
                        help="how many times to log training progress per epoch (default: 50)")
    parser.add_argument("--evaluations-per-epoch", type=int, default=50, metavar="N",
                        help="how many times to evaluate model per epoch (default: 50)")
    parser.add_argument("--reconstruction-saves-per-epoch", type=int, default=50, metavar="N",
                        help="how many batches to wait before running model on test images (default: 50)")

    parser.add_argument("--train-proportion", type=float, default=0.8, metavar="P",
                        help="proportion of data to be used for training (default: 0.8)")
    parser.add_argument("--validation-proportion", type=float, default=0.1,
                        help="proportion of data to be used for validation (default: 0.1)")
    parser.add_argument("--use-byol", action='store_true', default=False,
                        help="if present, uses BYOL model, otherwise SimCLR")
    parser.add_argument("--use-contrastive-output", action='store_true', default=False,
                        help="if present, uses embedding of vector as output. Otherwise, uses last set of feature maps.")
    parser.add_argument("--grayscale", action='store_true', default=False,
                        help="if present, output will be grayscale image.")
    parser.add_argument("--os", action='store_true', default=False,
                        help="if present, unet will be trained to generate canonical OS images.")
    parser.add_argument("--remove-copies", action='store_true', default=False,
                        help="if present, repeated OS maps won't be used.")


    # I/O params
    parser.add_argument("--patch-dataset-dir", required=True, help="Path to the data used to generate a patch dataset")
    parser.add_argument("--contrastive-checkpoint-dir", required=True, help="Path to the checkpoint containing the contrastive model")
    parser.add_argument("--experiment-name", required=True, help="Name of experiment")
    parser.add_argument("--input", required=True, help="Path to the input data for the model to read")

    parser.add_argument("--loss", choices=["MSE", "L1"],
                        help="type of loss to use (out of MSE and L1)")

    # debugging
    parser.add_argument("--debug", action='store_true', default=False,
                        help="disables model running to debug inputs")

    return parser

# --------------------------------------------------- MAIN ---------------------------------------------------

def main(args):

    torch.manual_seed(args.seed)

    PATCH_TRAIN_DATASET_DIR = os.path.join(args.patch_dataset_dir, f"patch_train_dataset_{args.patch_size}.pk")
    PATCH_VALIDATION_DATASET_DIR = os.path.join(args.patch_dataset_dir, f"patch_validation_dataset_{args.patch_size}.pk")
    PATCH_TEST_DATASET_DIR = os.path.join(args.patch_dataset_dir, f"patch_test_dataset_{args.patch_size}.pk")

    logging.info(f"File at {PATCH_TRAIN_DATASET_DIR}: {os.path.isfile(PATCH_TRAIN_DATASET_DIR)}")
    logging.info(f"File at {PATCH_VALIDATION_DATASET_DIR}: {os.path.isfile(PATCH_VALIDATION_DATASET_DIR)}")
    logging.info(f"File at {PATCH_TEST_DATASET_DIR}: {os.path.isfile(PATCH_TEST_DATASET_DIR)}")

    # create the CLPatchDataset object if it doesn't exist
    if not os.path.isfile(PATCH_TRAIN_DATASET_DIR) or not os.path.isfile(PATCH_VALIDATION_DATASET_DIR) or not os.path.isfile(PATCH_TEST_DATASET_DIR):
        patch_dataset = CLPatchDataset.from_dir(map_directory=args.input,
                                                patch_width=args.patch_size,
                                                verbose=True)

        patch_train_dataset, patch_validation_dataset, patch_test_dataset = patch_dataset.unique_split(p_train=args.train_proportion,
                                                                                                       p_validation=args.validation_proportion,
                                                                                                       seed=args.seed)

        patch_train_dataset.save(PATCH_TRAIN_DATASET_DIR)
        patch_validation_dataset.save(PATCH_VALIDATION_DATASET_DIR)
        patch_test_dataset.save(PATCH_TEST_DATASET_DIR)

    if args.os:
        logging.info("Using OS maps as canonical.")
        CANONICAL_TRAIN_DATASET_DIR = os.path.join(args.patch_dataset_dir,
                                                   f"canonical_os_train_dataset_{args.patch_size}.pk")
        CANONICAL_VALIDATION_DATASET_DIR = os.path.join(args.patch_dataset_dir,
                                                        f"canonical_os_validation_dataset_{args.patch_size}.pk")
        CANONICAL_TEST_DATASET_DIR = os.path.join(args.patch_dataset_dir,
                                                        f"canonical_os_test_dataset_{args.patch_size}.pk")

        logging.info(f"File at {CANONICAL_TRAIN_DATASET_DIR}: {os.path.isfile(CANONICAL_TRAIN_DATASET_DIR)}")
        logging.info(f"File at {CANONICAL_VALIDATION_DATASET_DIR}: {os.path.isfile(CANONICAL_VALIDATION_DATASET_DIR)}")
        logging.info(f"File at {CANONICAL_TEST_DATASET_DIR}: {os.path.isfile(CANONICAL_TEST_DATASET_DIR)}")

        # create the CanonicalDataset object (or load it if available)
        if os.path.isfile(CANONICAL_TRAIN_DATASET_DIR) and os.path.isfile(CANONICAL_VALIDATION_DATASET_DIR) and os.path.isfile(CANONICAL_TEST_DATASET_DIR):
            with open(CANONICAL_TRAIN_DATASET_DIR, "rb") as f:
                canonical_train_dataset = pk.load(f)

            with open(CANONICAL_VALIDATION_DATASET_DIR, "rb") as f:
                canonical_validation_dataset = pk.load(f)

            with open(CANONICAL_TEST_DATASET_DIR, "rb") as f:
                canonical_test_dataset = pk.load(f)
        else:

            with open(PATCH_TRAIN_DATASET_DIR, "rb") as f:
                patch_train_dataset = pk.load(f)

            with open(PATCH_VALIDATION_DATASET_DIR, "rb") as f:
                patch_validation_dataset = pk.load(f)

            with open(PATCH_TEST_DATASET_DIR, "rb") as f:
                patch_test_dataset = pk.load(f)

            canonical_patch_dict = CanonicalDataset.get_canonical_patch_dict_from_dataset(
                patch_datasets=[patch_train_dataset, patch_validation_dataset, patch_test_dataset],
                canonical_idx=1,
                data_dir=None)

            canonical_train_dataset = CanonicalDataset.from_os_dataset(patch_dataset = patch_train_dataset,
                                                                        canonical_idx = None,
                                                                        data_dir = None,
                                                                        canonical_patch_dict = canonical_patch_dict,
                                                                        remove_copies = args.remove_copies)
            canonical_validation_dataset = CanonicalDataset.from_os_dataset(patch_dataset = patch_validation_dataset,
                                                                            canonical_idx = None,
                                                                            data_dir = None,
                                                                            canonical_patch_dict = canonical_patch_dict,
                                                                            remove_copies = args.remove_copies)
            canonical_test_dataset = CanonicalDataset.from_os_dataset(patch_dataset=patch_test_dataset,
                                                                            canonical_idx=None,
                                                                            data_dir=None,
                                                                            canonical_patch_dict=canonical_patch_dict,
                                                                            remove_copies=args.remove_copies)

            canonical_train_dataset.save(CANONICAL_TRAIN_DATASET_DIR)
            canonical_validation_dataset.save(CANONICAL_VALIDATION_DATASET_DIR)
            canonical_test_dataset.save(CANONICAL_TEST_DATASET_DIR)
    else:

        logging.info("Using OSM maps as canonical.")

        CANONICAL_TRAIN_DATASET_DIR = os.path.join(args.patch_dataset_dir, f"canonical_train_dataset_{args.patch_size}.pk")
        CANONICAL_VALIDATION_DATASET_DIR = os.path.join(args.patch_dataset_dir, f"canonical_validation_dataset_{args.patch_size}.pk")
        CANONICAL_TEST_DATASET_DIR = os.path.join(args.patch_dataset_dir, f"canonical_test_dataset_{args.patch_size}.pk")

        logging.info(f"File at {CANONICAL_TRAIN_DATASET_DIR}: {os.path.isfile(CANONICAL_TRAIN_DATASET_DIR)}")
        logging.info(f"File at {CANONICAL_VALIDATION_DATASET_DIR}: {os.path.isfile(CANONICAL_VALIDATION_DATASET_DIR)}")
        logging.info(f"File at {CANONICAL_TEST_DATASET_DIR}: {os.path.isfile(CANONICAL_TEST_DATASET_DIR)}")

        # create the CanonicalDataset object (or load it if available)
        if os.path.isfile(CANONICAL_TRAIN_DATASET_DIR) and os.path.isfile(CANONICAL_VALIDATION_DATASET_DIR) and os.path.isfile(CANONICAL_TEST_DATASET_DIR):
            with open(CANONICAL_TRAIN_DATASET_DIR, "rb") as f:
                canonical_train_dataset = pk.load(f)

            with open(CANONICAL_VALIDATION_DATASET_DIR, "rb") as f:
                canonical_validation_dataset = pk.load(f)

            with open(CANONICAL_TEST_DATASET_DIR, "rb") as f:
                canonical_test_dataset = pk.load(f)
        else:
            canonical_train_dataset = CanonicalDataset.from_osm_dir(patch_dataset_dir = PATCH_TRAIN_DATASET_DIR,
                                                                    canonical_maps_dir= args.input)
            canonical_validation_dataset = CanonicalDataset.from_osm_dir(patch_dataset_dir=PATCH_VALIDATION_DATASET_DIR,
                                                                         canonical_maps_dir=args.input)
            canonical_test_dataset = CanonicalDataset.from_osm_dir(patch_dataset_dir=PATCH_TEST_DATASET_DIR,
                                                                   canonical_maps_dir=args.input)

            canonical_train_dataset.save(CANONICAL_TRAIN_DATASET_DIR)
            canonical_validation_dataset.save(CANONICAL_VALIDATION_DATASET_DIR)
            canonical_test_dataset.save(CANONICAL_TEST_DATASET_DIR)

    logging.info(f"Generated training dataset with {len(canonical_train_dataset)} samples.")
    logging.info(f"Generated validation dataset with {len(canonical_validation_dataset)} samples.")
    logging.info(f"Generated validation dataset with {len(canonical_test_dataset)} samples.")

    # create the DataLoader object
    canonical_train_loader = DataLoader(canonical_train_dataset,
                                        batch_size = args.batch_size,
                                        shuffle = True,
                                        num_workers = 2)
    canonical_validation_loader = DataLoader(canonical_validation_dataset,
                                             batch_size=args.batch_size,
                                             shuffle = False,
                                             num_workers=2)

    unet = UNet.from_checkpoint(checkpoint_dir = args.contrastive_checkpoint_dir,
                                model = MapBYOL if args.use_byol else MapSIMCLR,
                                model_kwargs=None, use_resnet=True,
                                use_contrastive_output=args.use_contrastive_output,
                                grayscale_output=args.grayscale)

    #idxs = [3, 748, 9287, 198080, 57]
    idxs = [3, 57, 5381, 283, 23]
    historical_imgs = [torch.Tensor(canonical_train_dataset.historical_patches[i].patch) for i in idxs]
    historical_imgs = torch.stack(historical_imgs)
    canonical_imgs = [torch.Tensor(canonical_train_dataset.canonical_patches[i].patch) for i in idxs]
    canonical_imgs = torch.stack(canonical_imgs)

    unet.compile_model(historical_imgs = historical_imgs,
                       canonical_imgs = canonical_imgs,
                       loss_str = args.loss,
                       optimiser = optim.Adam,
                       lr = args.lr)

    checkpoint_parent = os.path.abspath(os.path.join(args.contrastive_checkpoint_dir, os.pardir))
    canonical_checkpoint_dir = os.path.join(checkpoint_parent, args.experiment_name)

    logging.info(f"Saving checkpoints at {canonical_checkpoint_dir}")
    logging.info(f"Using device: {unet.device}")

    torch.cuda.empty_cache()

    # train the model
    checkpoint = unet.train_model(train_loader = canonical_train_loader,
                                  validation_loader = canonical_validation_loader,
                                  epochs = args.epochs,
                                  checkpoint_dir = canonical_checkpoint_dir,
                                  logs_per_epoch = args.logs_per_epoch,
                                  evaluations_per_epoch = args.evaluations_per_epoch,
                                  reconstruction_saves_per_epoch= args.reconstruction_saves_per_epoch)

# --------------------------------------------------- RUN ---------------------------------------------------

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")
    if not args.debug:
        main(args)
    else:
        print(args)

