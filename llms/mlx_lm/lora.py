# Copyright © 2024 Apple Inc.

import argparse
import math
import re
import types
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten

from .tuner.datasets import load_dataset
from .tuner.trainer import TrainingArgs, TrainingCallback, evaluate, train, evaluate_test
from .tuner.utils import build_schedule, linear_to_lora_layers
from .utils import load, save_config

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


CONFIG_DEFAULTS = {
    "model": "mlx_model",
    "train": False,
    "data": "data/",
    "seed": 0,
    "lora_layers": 16,
    "batch_size": 4,
    "iters": 1000,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "lr_schedule": None,
    "base_model": False,
    "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},
}


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--model",
        help="The path to the local model directory or Hugging Face repo.",
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, help="Minibatch size.")
    parser.add_argument("--iters", type=int, help="Iterations to train for.")
    parser.add_argument(
        "--val-batches",
        type=int,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument(
        "--steps-per-report",
        type=int,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        help="Load path to resume training with the given adapters.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Save/load path for the adapters.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="A YAML configuration file with the training options",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to reduce memory use.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    
    parser.add_argument("--base_model", type=bool, default=False, help="For not use adapters")
    return parser


def print_trainable_parameters(model):
    def nparams(m):
        if isinstance(m, nn.QuantizedLinear):
            return m.weight.size * (32 // m.bits)
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )
    print(
        f"Trainable parameters: {(trainable_p * 100 / total_p):.3f}% "
        f"({trainable_p:.3f}M/{total_p:.3f}M)"
    )

from pathlib import Path

def ensure_directory_exists(dir_path):
    """
    Verifica si un directorio existe y lo crea si no es así usando pathlib.
    
    Args:
    dir_path (str or Path): Ruta del directorio a verificar y crear.
    """
    path = Path(dir_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directorio '{path}' creado.")
    else:
        print(f"Directorio '{path}' ya existe.")

def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load(args.model)

    # Freeze all layers
    model.freeze()
    # Convert linear layers to lora layers and unfreeze in the process
    linear_to_lora_layers(model, args.lora_layers, args.lora_parameters)

    print_trainable_parameters(model)

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    # Resume training the given adapters.
    if args.resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    if args.train:
        adapter_path = Path(args.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)

        # Crear el segundo directorio con el sufijo "_best_val"
        # Utilizamos 'with_name' para cambiar el nombre del directorio final añadiendo el sufijo
        best_val_directory_path = adapter_path.with_name(adapter_path.name + "_best_val")

        # Asegurar que el directorio padre del nuevo directorio existe
        best_val_directory_path.parent.mkdir(parents=True, exist_ok=True)
        # Crear el directorio con sufijo si no existe
        best_val_directory_path.mkdir(exist_ok=True)

        save_config(vars(args), adapter_path / "adapter_config.json")
        adapter_file = adapter_path / "adapters.safetensors"
        print("Training")
        
        # init training args
        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
        )

        model.train()
        opt = optim.Adam(
            learning_rate=(
                build_schedule(args.lr_schedule)
                if args.lr_schedule
                else args.learning_rate
            )
        )
        # import pdb
        # pdb.set_trace()
        # Train model
        train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            training_callback=training_callback,
        )
    if args.base_model == False:
        # Load the LoRA adapter weights which we assume should exist by this point
        adapter_file ='adapters/adapters.safetensors'
        #adapter_file ='adapters_best_val/adapters.safetensors'
        
        if not Path(adapter_file).is_file():
            raise ValueError(
                f"Adapter file {adapter_file} missing. "
                "Use --train to learn and save the adapters"
            )
        model.load_weights(str(adapter_file), strict=False)

    if args.test:
        print("Testing")
        model.eval()
        
        test_loss = evaluate_test(
            model=model,
            dataset=test_set,
            tokenizer=tokenizer,
            prefix="test"
        )
        
        # val_loss = evaluate_test(
        #     model=model,
        #     dataset=valid_set,
        #     tokenizer=tokenizer,
        #     prefix="val"
        # )
        # train_loss = evaluate_test(
        #     model=model,
        #     dataset=train_set,
        #     tokenizer=tokenizer,
        #     prefix="train"
        # )

        
        # To save ppl results
        
        # train_ppl = math.exp(train_loss)
        # print(f"Test loss {train_ppl:.3f}, Test ppl {train_ppl:.3f}.")
        
        # losses = {
        # 'test': test_loss,
        # 'val': val_loss,
        # 'train': train_loss
        # }

        # perplexities = {
        #     'test': test_ppl,
        #     'val': val_ppl,
        #     'train': train_ppl
        # }
        # import csv
        
        # for set in ["test", "val", "train"]:
        #     with open(f'{set}.csv', 'w', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([f'{set} loss', f'{set} ppl'])
        #         # Use the dictionary to get the correct loss and perplexity for each set
        #         writer.writerow([f"{losses[set]:.3f}", f"{perplexities[set]:.3f}"])

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    config = args.config
    args = vars(args)
    if config:
        print("Loading configuration file", config)
        with open(config, "r") as file:
            config = yaml.load(file, yaml_loader)
        # Prefer parameters from command-line arguments
        for k, v in config.items():
            if not args.get(k, None):
                args[k] = v

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if not args.get(k, None):
            args[k] = v
    run(types.SimpleNamespace(**args))

