import argparse
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(description="Generate experiments for SLURM jobs")

    parser.add_argument("--main", required=True,
                        help="path to the main.py to be run by SLURM.")
    parser.add_argument("--scratch-data-dir", required=True,
                        help="path to the data in SLURM for training.")
    parser.add_argument("--scratch-out-dir", required=True,
                        help="path to the output folder in SLURM during training.")
    parser.add_argument("--patch-dataset-dir", required=True, help="Path to the data used to generate a patch dataset")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Path to the checkpoint containing the contrastive model")

    return parser


def get_default_arg_dict(scratch_data_dir, scratch_out_dir, patch_data_dir, checkpoint_dir):
    return {
        "--batch-size": 32,
        "--patch-size": 64,
        "--epochs": 1,
        "--lr" : 1e-3,
        "--seed": 23,
        "--log-interval": 1000,
        "--save-reconstruction-interval": 250,
        "--train-proportion": 0.98,
        "--input": scratch_data_dir,
        "--output": scratch_out_dir,
        "--patch-dataset-dir": patch_data_dir,
        "--checkpoint-dir": checkpoint_dir,
        "--use-byol": True,
        "--use-contrastive-output": True,
        "--loss" : "MSE"
    }


def create_experiment(main_file, scratch_data_dir, scratch_out_dir, patch_data_dir, checkpoint_dir, experiment_args):
    arg_dict = get_default_arg_dict(scratch_data_dir, scratch_out_dir, patch_data_dir, checkpoint_dir)

    for arg, value in experiment_args.items():
        if arg in arg_dict:
            if type(value) == bool:
                if not value:
                    arg_dict.pop(arg)
            else:
                arg_dict[arg] = value
        else:
            raise ValueError(f"The provided argument {arg} isn't a valid experiment argument.")

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

    experiment_argss = [{"--epochs": 5, "--encoder": "cnn", "--encoder-layer-idx": -1, "--patch-size": 128}]

    experiment_run_args = [create_experiment(main_file=args.main,
                                             scratch_data_dir=args.scratch_data_dir,
                                             scratch_out_dir=args.scratch_out_dir,
                                             patch_data_dir=args.patch_data_dir,
                                             checkpoint_dir=args.checkpoint_dir,
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
