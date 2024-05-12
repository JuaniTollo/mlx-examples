# Copyright © 2024 Apple Inc.

import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
import time
import pandas as pd
from ..utils import load, save_config

import pdb

def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


@dataclass
class TrainingArgs:
    lora_layers: int = field(
        default=16, metadata={"help": "Number of layers to fine-tune"}
    )
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )


def default_loss(model, inputs, targets, lengths):
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


def iterate_batches(dataset, tokenizer, batch_size, max_seq_length, train=False):
    # Sort by length:
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]))

    # Make the batches:
    batch_idx = [
        idx[i : i + batch_size] for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            # Encode batch
            batch = [tokenizer.encode(dataset[j]) for j in batch_idx[i]]

            lengths = [len(x) for x in batch]

            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the nearest multiple of 8 or the maximum length
            pad_to = 8
            max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size, max_length_in_batch), np.int32)

            for j in range(batch_size):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )
            batch = mx.array(batch_arr)

            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(
    model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


class TrainingCallback:

    def on_train_loss_report(self, train_info: dict):
        """Called to report training loss at specified intervals."""
        pass

    def on_val_loss_report(self, val_info: dict):
        """Called to report validation loss at specified intervals or the beginning."""
        pass


def train(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    
    print(f"Starting training..., iters: {args.iters}")
    
    data_loss = []
    data_val = []
    best_val_loss = 999999
    
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)

        return lvalue, toks

    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0
    trained_tokens = 0
    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        lvalue, toks = step(batch)
        mx.eval(state, lvalue, toks)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = np.mean(losses)
            # Nueva fila a insertar

            data_loss.append([it,train_loss])

            stop = time.perf_counter()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.metal.get_peak_memory() / 2**30
            print(
                f"Iter {it}: Train loss {train_loss:.3f}, "
                f"Learning Rate {learning_rate:.3e}, "
                f"It/sec {it_sec:.3f}, "
                f"Tokens/sec {tokens_sec:.3f}, "
                f"Trained Tokens {trained_tokens}, "
                f"Peak mem {peak_mem:.3f} GB"
            )
                        
            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            val_time = time.perf_counter() - stop
            print(
                f"Iter {it}: " f"Val loss {val_loss:.3f}, " f"Val took {val_time:.3f}s"
            )
            
            data_val.append([it, val_loss])

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            start = time.perf_counter()
                
        # Save adapter weights
        if it % args.steps_per_save == 0:
            save_adapter(model, args.adapter_file)
            checkpoint = (
                
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            save_adapter(model, checkpoint)
            
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                
                # Assuming args.adapter_file is the full path to the current adapter file including filename
                adapter_file_path = Path(args.adapter_file)
                
                # Create a new directory path by appending "_best_val" to the parent directory of the adapter file
                best_val_directory = (adapter_file_path.parent.name + "_best_val")

                # Define the new file path within the new directory with the same file name as the original adapter file
                best_val_file_path = best_val_directory +"/"+ adapter_file_path.name

                checkpoint = (
                Path(best_val_file_path).parent / f"{it:07d}_adapters.safetensors"
            )
                # Save the adapter to this new path
                save_adapter(model, best_val_file_path)
                save_adapter(model, checkpoint)

                # Print confirmation
                print(f"Best validation model saved to {best_val_file_path}")
            # Intended cooldown period (no actual delay)
            time.sleep(30)
            
    # from pathlib import Path    
    # # Guardar el tokenizer en el directorio especificado
    # tokenizer_save_path =Path("tokenizer")
    # tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    # tokenizer.save_pretrained(tokenizer_save_path)
    
    # print(f"Tokenizer saved to {tokenizer_save_path}")         

    # Crear DataFrame al final
    df_loss = pd.DataFrame(data_loss, columns=['Iteration','Loss'])
    df_loss.to_csv("Train_loss.csv", index=False)

        # Crear DataFrame al final
    df_val = pd.DataFrame(data_val, columns=['Iteration','Loss'])
    df_val.to_csv("Val_loss.csv", index=False)

    # save final adapter weights
    save_adapter(model, args.adapter_file)
    print(f"Saved final adapter weights to {args.adapter_file}.")


def save_adapter(
    model: nn.Module,
    adapter_file: Union[str, Path],
):
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(adapter_file), dict(flattened_tree))

def devolverUltimaPosicionNoNulaTargets(array):
    # Inicializa una lista para almacenar los valores deseados
    valores_finales_no_cero = []
    indices = []
    # Itera sobre cada uno de los vectores en el array
    for vector in array:
        vector = np.array(vector)
        # Filtra el vector para quedarte solo con los valores no cero
        valores_no_cero = vector[vector != 0]
        indices_no_cero = np.nonzero(vector)[0]

        # Obtiene el último valor no cero, si existe
        if valores_no_cero.size > 0:
            ultimo_valor_no_cero = valores_no_cero[-1]
        else:
            ultimo_valor_no_cero = None  # O lo que consideres apropiado para indicar que no hay valores no cero
        # Agrega el valor a la lista
        valores_finales_no_cero.append(ultimo_valor_no_cero)
        indices.append(int(indices_no_cero[-1]))
    indice =  ultimo_valor_no_cero
    # Muestra los resultados
    return (np.array(indices)).tolist()

def loss_test(model, tokenizer, inputs, targets, lengths):
    pdb.set_trace()

    logits, _ = model(inputs)
    
    logits = logits.astype(mx.float32)
    
    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks, logits

import glob
import os 
import gc  # Garbage collector interface
from tqdm import tqdm  # Importa tqdm
import pdb

def concatenate_and_cleanup(prefix):
    # Concatenar y guardar los arrays de targets
    all_targets = []
    files_targets = glob.glob(f'./{prefix}_targets_global_*.npy')
    for file in sorted(files_targets, key=lambda x: int(x.split('_')[-1].split('.')[0])):
        all_targets.append(np.load(file))
    all_targets = np.concatenate(all_targets)
    np.save(f'./{prefix}_all_targets.npy', all_targets)

    # Concatenar y guardar los arrays de logits
    all_logits = []
    files_logits = glob.glob(f'./{prefix}_logits_global_*.npy')
    for file in sorted(files_logits, key=lambda x: int(x.split('_')[-1].split('.')[0])):
        all_logits.append(np.load(file))
    all_logits = np.concatenate(all_logits, axis=0)
    np.save(f'./{prefix}_all_logits.npy', all_logits)

    # Eliminar los archivos individuales
    for file in files_targets + files_logits:
        os.remove(file)

    print("Concatenation and cleanup completed.")



def evaluate_test(model, dataset, tokenizer, prefix, max_seq_length=2048):
    
    all_losses = []
    ntokens = 0
    targets_global = []
    logits_global = []
    
    index_iterator = iter(range(len(dataset))) if len(dataset) != -1 else iter(int, 1)
    
    i = 0
    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=1,
            max_seq_length=max_seq_length,
        ),
    ):

        losses, toks, targets, logits = loss_test(model,tokenizer, *batch)
        
        all_losses.append((losses * toks).item())
        ntokens += toks.item()
        i +=1
        
        #last_tkn_idx = int(np.nonzero(targets[0])[0][-1])
        indices = devolverUltimaPosicionNoNulaTargets(indices)
        
        # Generar los índices para la primera dimensión
        first_dim_indices = np.arange(targets.shape[0]).tolist()
        
        targets_global.append(np.array(targets)[first_dim_indices, indices])

        first_dim_logits = np.arange(targets.shape[0]).tolist()        
        logits_global.append(np.array(logits)[first_dim_logits, indices, :])

        if i % 2 == 0:
            print(i)
            
            np.save(f'./{prefix}_targets_global_{i}.npy', (targets_global))
            print(targets_global[0].shape, "#targets",len(targets_global))
            
            np.save(f'./{prefix}_logits_global_{i}.npy', logits_global)
            print(logits_global[0].shape, "#logits", len(logits_global))
            
            targets_global = []
            logits_global = []
            gc.collect()
            #time.sleep(5)
            
    concatenate_and_cleanup(prefix)
            
    # Guardar el tokenizer en el directorio especificado
    tokenizer_save_path = Path("tokenizer")
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    # Use save_pretrained method to save the tokenizer
    tokenizer.save_pretrained(tokenizer_save_path)
    
    return np.sum(all_losses) / ntokens

