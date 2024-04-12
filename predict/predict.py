# script que voy a llamar cuando haga el experimento
# evaluar
# Recibe argumentos. Base de datos. Tamano del barch.
# al main le paso el config.
#python predict.py --adaptation lora


#Cantidad instancias, tipo de base de datos, modelo base o método de adaptación.
import argparse
import sys
import sys
sys.path.insert(0, '../Lora')

from lora import *

def build_parser():
    parser = argparse.ArgumentParser(description="Predicting logits for different fine tunning strategies")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adaptation",
        default="lora",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )

    # Add the 'train' argument
    parser.add_argument(
        "--train",
        action="store_true",
        default=True,  # Set default to True
        help="We need this to be true in order to use load at lora.py."
    )  
    # Add the 'train' argument
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,  # Set default to True
        help="We need this to be true in order to use load at lora.py"
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    print(args.adaptation)
    print(args.data)
    from pathlib import Path

    print(Path(args.data))
    if args.adaptation == "lora":
        train, valid, test = load(args)

    
    model.load_weights(args.adapter_file, strict=False)

    if args.test:
        print("Testing")
        model.eval()
        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args.batch_size,
            num_batches=args.test_batches,
        )
        test_ppl = math.exp(test_loss)
        
#Esto lo voy a necesitar para Lora.
    # parser.add_argument(
    #     "--adapter-file",
    #     type=str,
    #     default="adapters.npz",
    #     help="Save/load path for the trained adapter weights.",
    # )
