import argparse
from datetime import datetime
import os


def get_parser():
    parser = argparse.ArgumentParser(description="Generate experiments for SLURM jobs")

    parser.add_argument("--main", required=True,
                        help="path to the main.py to be run by SLURM.")
    parser.add_argument("--scratch-data-dir", required=True,
                        help="path to the data in SLURM for training.")
    parser.add_argument("--scratch-out-dir", required=True,
                        help="path to the output folder in SLURM during training.")
    parser.add_argument("--patch-dataset-dir", required=True, help="Path to the data used to generate a patch dataset")

    return parser


def get_default_arg_dict(scratch_data_dir, patch_data_dir):
    return {
        "--batch-size": 32,
        "--patch-size": 64,
        "--epochs": 1,
        "--lr" : 1e-3,
        "--seed": 23,
        "--logs-per-epoch": 50,
        "--evaluations-per-epoch": 150,
        "--reconstruction-saves-per-epoch": 4,
        "--train-proportion": 0.8,
        "--validation-proportion": 0.1,
        "--input": scratch_data_dir,
        "--patch-dataset-dir": patch_data_dir,
        "--contrastive-checkpoint-dir": None,
        "--experiment-name": None,
        "--use-byol": True,
        "--use-contrastive-output": True,
        "--loss": "MSE",
        "--grayscale": False,
        "--os": False,
        "--remove-copies": False
    }

def get_experiment_name(epochs, batch_size, patch_size, lr, use_contrastive_output, loss, grayscale, os, remove_copies):
    string_lr = str(lr).replace(".", "_").replace(",", "_")
    return f"can{'g' if grayscale else ''}-{'rc-' if remove_copies else ''}{'os-' if os else ''}{'co' if use_contrastive_output else 'nco'}-{loss}-e{epochs}-lr{string_lr}-b{batch_size}-p{patch_size}"

def create_experiment(main_file, scratch_data_dir, patch_dataset_dir, experiment_args):
    arg_dict = get_default_arg_dict(scratch_data_dir, patch_dataset_dir)

    for arg, value in experiment_args.items():
        if arg in arg_dict:
            if type(value) == bool:
                if not value:
                    arg_dict.pop(arg)
                else:
                    arg_dict[arg] = value
            else:
                arg_dict[arg] = value
        else:
            raise ValueError(f"The provided argument {arg} isn't a valid experiment argument.")

    if arg_dict["--contrastive-checkpoint-dir"] is None:
        raise ValueError(f"contrastive-checkpoint-dir key in argument dict is mandatory; a contrastive model is required.")

    arg_dict["--experiment-name"] = get_experiment_name(epochs=arg_dict["--epochs"],
                                                        batch_size=arg_dict["--batch-size"],
                                                        lr=arg_dict["--lr"],
                                                        patch_size=arg_dict["--patch-size"],
                                                        use_contrastive_output=arg_dict.get("--use-contrastive-output", False),
                                                        loss=arg_dict["--loss"],
                                                        grayscale=arg_dict.get("--grayscale", False),
                                                        os=arg_dict.get("--os", False),
                                                        remove_copies=arg_dict.get("--remove-copies", False))

    python_call = f"python {main_file}"

    for arg, value in arg_dict.items():
        if type(value) == str:
            python_call += f' {arg} "{value}"'
        elif type(value) == bool:
            python_call += f' {arg}'
        else:
            python_call += f' {arg} {value}'

    return python_call


def main(args):
    # testing: high epochs, simclr & byol, os vs osm

    EPOCHS = 20
    RECONSTRUCTION_SAVES_PER_EPOCH = 4
    TRAIN_PROPORTION = 0.8
    VALIDATION_PROPORTION = 0.1

    experiment_argss = [
        # simclr, not contrastive output, OSM
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--lr": 1e-3,
            "--seed": 23,
            "--logs-per-epoch": 50,
            "--evaluations-per-epoch": 150,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": VALIDATION_PROPORTION,
            "--reconstruction-saves-per-epoch": RECONSTRUCTION_SAVES_PER_EPOCH,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-cnn-e25-b64-t0_99-lr0_001-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": False,
            "--remove-copies": False
        },
        # byol, not contrastive output, OSM
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--lr": 1e-3,
            "--seed": 23,
            "--logs-per-epoch": 50,
            "--evaluations-per-epoch": 150,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": VALIDATION_PROPORTION,
            "--reconstruction-saves-per-epoch": RECONSTRUCTION_SAVES_PER_EPOCH,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "b-presnet18-e25-b64-t0_80-lr0_001-p128",
                                                         "byol_checkpoint.pt"),
            "--use-byol": True,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": False,
            "--remove-copies": False
        },
        # simclr, not contrastive output, OS
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--lr": 1e-3,
            "--seed": 23,
            "--logs-per-epoch": 50,
            "--evaluations-per-epoch": 150,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": VALIDATION_PROPORTION,
            "--reconstruction-saves-per-epoch": RECONSTRUCTION_SAVES_PER_EPOCH,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-cnn-e25-b64-t0_99-lr0_001-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": True,
            "--remove-copies": False
        },
        # simclr, not contrastive output, OS
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--lr": 1e-3,
            "--seed": 23,
            "--logs-per-epoch": 50,
            "--evaluations-per-epoch": 150,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": VALIDATION_PROPORTION,
            "--reconstruction-saves-per-epoch": RECONSTRUCTION_SAVES_PER_EPOCH,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "b-presnet18-e25-b64-t0_80-lr0_001-p128",
                                                         "byol_checkpoint.pt"),
            "--use-byol": True,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": True,
            "--remove-copies": False
        },
        # simclr, contrastive output, OSM
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--lr": 1e-3,
            "--seed": 23,
            "--logs-per-epoch": 50,
            "--evaluations-per-epoch": 150,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": VALIDATION_PROPORTION,
            "--reconstruction-saves-per-epoch": RECONSTRUCTION_SAVES_PER_EPOCH,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-cnn-e25-b64-t0_99-lr0_001-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": True,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": False,
            "--remove-copies": False
        },
        # byol, contrastive output, OSM
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--lr": 1e-3,
            "--seed": 23,
            "--logs-per-epoch": 50,
            "--evaluations-per-epoch": 150,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": VALIDATION_PROPORTION,
            "--reconstruction-saves-per-epoch": RECONSTRUCTION_SAVES_PER_EPOCH,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "b-presnet18-e25-b64-t0_80-lr0_001-p128",
                                                         "byol_checkpoint.pt"),
            "--use-byol": True,
            "--use-contrastive-output": True,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": False,
            "--remove-copies": False
        },
        # simclr, contrastive output, OS
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--lr": 1e-3,
            "--seed": 23,
            "--logs-per-epoch": 50,
            "--evaluations-per-epoch": 150,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": VALIDATION_PROPORTION,
            "--reconstruction-saves-per-epoch": RECONSTRUCTION_SAVES_PER_EPOCH,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-cnn-e25-b64-t0_99-lr0_001-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": True,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": True,
            "--remove-copies": False
        },
        # byol, contrastive output, OS
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--lr": 1e-3,
            "--seed": 23,
            "--logs-per-epoch": 50,
            "--evaluations-per-epoch": 150,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": VALIDATION_PROPORTION,
            "--reconstruction-saves-per-epoch": RECONSTRUCTION_SAVES_PER_EPOCH,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "b-presnet18-e25-b64-t0_80-lr0_001-p128",
                                                         "byol_checkpoint.pt"),
            "--use-byol": True,
            "--use-contrastive-output": True,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": True,
            "--remove-copies": False
        }
    ]
    """
    experiment_argss = [
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": 30,
            "--lr": 1e-3,
            "--seed": 23,
            "--log-interval": 50,
            "--reconstruction-saves-per-epoch": 1000,
            "--train-proportion": 0.98,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-presnet18-e5-b32-t0_99-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": False,
            "--os" : False,
            "--remove-copies" : False
        },
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": 30,
            "--lr": 1e-3,
            "--seed": 23,
            "--log-interval": 50,
            "--reconstruction-saves-per-epoch": 1000,
            "--train-proportion": 0.98,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "b-presnet18-e5-b32-t0_9-p128",
                                                         "byol_checkpoint.pt"),
            "--use-byol": True,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": False,
            "--remove-copies": False
        },
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": 30,
            "--lr": 1e-3,
            "--seed": 23,
            "--log-interval": 50,
            "--reconstruction-saves-per-epoch": 1000,
            "--train-proportion": 0.98,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-presnet18-e5-b32-t0_99-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": False,
            "--os": True,
            "--remove-copies": True
        }
        ]
    """
    #TODO
    # - change learning rate
    # - grayscale vs no grayscale
    # - different BYOL
    """
    experiment_argss = [
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": 5,
            "--lr": 1e-3,
            "--seed": 23,
            "--log-interval": 250,
            "--reconstruction-saves-per-epoch": 250,
            "--train-proportion": 0.98,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-presnet18-e5-b32-t0_99-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": False,
        },
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": 5,
            "--lr": 1e-3,
            "--seed": 23,
            "--log-interval": 250,
            "--reconstruction-saves-per-epoch": 250,
            "--train-proportion": 0.98,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-presnet18-e5-b32-t0_99-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": False,
            "--loss": "MSE",
            "--grayscale": True
        },
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": 5,
            "--lr": 1e-3,
            "--seed": 23,
            "--log-interval": 250,
            "--reconstruction-saves-per-epoch": 250,
            "--train-proportion": 0.98,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                                         "s-presnet18-e5-b32-t0_99-p128",
                                                         "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": True,
            "--loss": "MSE",
            "--grayscale": True
        }
    ]


    experiment_argss = [
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": 5,
            "--lr": 1e-3,
            "--seed": 23,
            "--log-interval": 500,
            "--reconstruction-saves-per-epoch": 250,
            "--train-proportion": 0.98,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                             "s-presnet18-e5-b32-t0_99-p128",
                                             "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": False,
            "--loss": "MSE"
        },
        {
            "--batch-size": 64,
            "--patch-size": 128,
            "--epochs": 5,
            "--lr": 1e-3,
            "--seed": 23,
            "--log-interval": 500,
            "--reconstruction-saves-per-epoch": 250,
            "--train-proportion": 0.98,
            "--contrastive-checkpoint-dir": os.path.join(args.scratch_out_dir,
                                             "s-presnet18-e5-b32-t0_99-p128",
                                             "simclr_checkpoint.pt"),
            "--use-byol": False,
            "--use-contrastive-output": True,
            "--loss": "MSE"
        }
    ]
    """

    experiment_run_args = [create_experiment(main_file=args.main,
                                             scratch_data_dir=args.scratch_data_dir,
                                             patch_dataset_dir=args.patch_dataset_dir,
                                             experiment_args=experiment_args) + "\n"
                           for experiment_args in experiment_argss]

    with open("canonical_experiments.txt", "w+") as f:
        f.writelines(experiment_run_args)

    with open("ran_canonical_experiments.txt", "a+") as f:
        f.write("\n")
        f.write("=" * 20 + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "=" * 20 + "\n")
        f.writelines(experiment_run_args)
        f.write("=" * 60)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
