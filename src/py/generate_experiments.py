import argparse
from datetime import datetime

def get_parser():
	parser = argparse.ArgumentParser(description="Generate experiments for SLURM jobs")

	parser.add_argument("-i", "--main", required=True,
		help="path to the main.py to be run by SLURM.")
	parser.add_argument("-d", "--scratch-data-dir", required=True,
						help="path to the data in SLURM for training.")
	parser.add_argument("-o", "--scratch-out-dir", required=True,
						help="path to the output folder in SLURM during training.")

	return parser

def get_experiment_name(use_byol, encoder, epochs, batch_size, tau, patch_size, pretrained):
	string_tau = str(tau).replace(".", "_").replace(",", "_")
	return f"{'b' if use_byol else 's'}-{'p' if pretrained else ''}{encoder}-e{epochs}-b{batch_size}-t{string_tau}-p{patch_size}"

def get_default_arg_dict(scratch_data_dir, scratch_out_dir):
	return {
	"--batch-size" : 32,
	"--patch-size" : 64,
	"--epochs" : 1,
	"--seed" : 23,
	"--log-interval" : 1000,
	"--train-proportion" : 0.98,
	"--input" : scratch_data_dir,
	"--output" : scratch_out_dir,
	"--experiment-name" : "b-r18-e1-b64-t0_99-p64",
	"--use-byol" : True,
	"--encoder" : "resnet18",
	"--pretrain-encoder" : True, 
	"--encoder-layer-idx" : -2,
	"--byol-ema-tau" : 0.99,
	"--simclr-tau" : 0.99
	}

def create_experiment(main_file, scratch_data_dir, scratch_out_dir, experiment_args):

	arg_dict = get_default_arg_dict(scratch_data_dir, scratch_out_dir)

	for arg, value in experiment_args.items():
		if arg in arg_dict:
			if type(value) == bool:
				if not value:
					arg_dict.pop(arg) 
			else:
				arg_dict[arg] = value
		else:
			raise ValueError(f"The provided argument {arg} isn't a valid experiment argument.")

	arg_dict["--experiment-name"] = get_experiment_name(use_byol = arg_dict.get("--use-byol", False),
														encoder = arg_dict["--encoder"],
														epochs = arg_dict["--epochs"],
														batch_size = arg_dict["--batch-size"],
														tau = arg_dict["--byol-ema-tau" if arg_dict.get("--use-byol", False) else "--simclr-tau"],
														patch_size = arg_dict["--patch-size"],
														pretrained = arg_dict.get("--pretrain-encoder", False))

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

	# current run
	# INFO: Using BYOL, but with different patch sizes to see whether it has any effect on learning
	"""
        experiment_argss = [{"--epochs": 5, "--patch-size": 64, "--batch-size" : 32},
						{"--epochs": 5, "--patch-size": 32, "--batch-size" : 32},
						{"--epochs": 5, "--patch-size": 128, "--batch-size" : 32}
						]
        """
	# previous runs
	experiment_argss = [{"--epochs": 5, "--use-byol" : False, "--encoder-layer-idx": -2},
						{"--epochs": 5, "--pretrain-encoder": False},
						{"--epochs": 5, "--pretrain-encoder": False, "--encoder": "resnet34"},
						{"--epochs": 5, "--pretrain-encoder": False, "--encoder": "cnn", "--encoder-layer-idx" : -1},
						{"--epochs": 5, "--encoder": "resnet34"},
						{"--epochs": 5, "--byol-ema-tau": 0.95},
						{"--epochs": 5, "--byol-ema-tau": 0.9},
						{"--epochs": 5, "--byol-ema-tau": 0.8}
						]



	experiment_run_args = [create_experiment(main_file = args.main,
											 scratch_data_dir = args.scratch_data_dir,
											 scratch_out_dir = args.scratch_out_dir,
											 experiment_args = experiment_args) + "\n"
						   for experiment_args in experiment_argss]

	with open("experiments.txt", "w+") as f:
		f.writelines(experiment_run_args)

	with open("ran_experiments.txt", "a+") as f:
		f.write("\n")
		f.write("="*20 + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "="*20 + "\n")
		f.writelines(experiment_run_args)
		f.write("=" * 60)


if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
	main(args)
