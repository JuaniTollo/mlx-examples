import json
import random
from typing import List, Tuple

def read_jsonl(file_path: str) -> List[dict]:
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def shuffle_data(data: List[dict]) -> None:
    """Shuffle the data in place."""
    random.shuffle(data)

def split_data(data: List[dict], test_size: float) -> Tuple[List[dict], List[dict]]:
    """Split the data into training and test sets based on the test size percentage."""
    split_index = int(len(data) * test_size)
    return data[split_index:], data[:split_index]

def save_jsonl(data: List[dict], file_path: str) -> None:
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')
import os
print("Current working directory:", os.getcwd())

def main(train_file_path: str, test_size: float):
    print("Reading data from:", train_file_path)  # This line will confirm the path being accessed
    data = read_jsonl(train_file_path)
    shuffle_data(data)
    train_data, test_data = split_data(data, test_size)
    save_jsonl(train_data, 'data/train.jsonl')
    save_jsonl(test_data, 'data/test.jsonl')

train_file_path = "./data/train.jsonl"
main(train_file_path, 0.1)