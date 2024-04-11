import csv
import json
import os

instruction_prompt = "Read the scenario and question below, then choose the most appropriate answer from the options given (1, 2, or 3). Respond with the number of the correct answer based on the scenario's context and the question's implications."

def convert_csv_to_jsonl(input_csv_path, output_jsonl_path):
    # Define a function to format the input text
    def format_input(row):
        input_text = row["inputs"].replace('[socialiqa]:\n', '').strip()
        input_text = input_text.replace('<context>', '\nC ').replace('</context>', ' ')
        input_text = input_text.replace('<question>', '\nQ ').replace('</question>', ' ')
        input_text = input_text.replace('\nC ', f'\nC {instruction_prompt}')
        input_text += f" <answerA>{row['inputs'].split('<answerA>')[1].split('</answerA>')[0]}</answerA>"
        input_text += f" <answerB>{row['inputs'].split('<answerB>')[1].split('</answerB>')[0]}</answerB>"
        input_text += f" <answerC>{row['inputs'].split('<answerC>')[1].split('</answerC>')[0]}</answerC>"
        input_text += f"\nA {row['targets']}"
        
        # Perform replacements
        input_text = input_text.replace("answerA", "answer1")
        input_text = input_text.replace("answerB", "answer2")
        input_text = input_text.replace("answerC", "answer3")

        return input_text

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    # Read the CSV file and write to JSONL
    with open(input_csv_path, mode='r', encoding='utf-8') as csv_file, open(output_jsonl_path, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            formatted_text = format_input(row)
            json.dump({"text": formatted_text}, jsonl_file)
            jsonl_file.write('\n')

def process_directory(input_dir, output_dir):
    # List all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_csv_path = os.path.join(input_dir, filename)
            output_jsonl_path = os.path.join(output_dir, filename.replace('.csv', '.jsonl'))
            convert_csv_to_jsonl(input_csv_path, output_jsonl_path)
            print(f'Processed {input_csv_path} and saved to {output_jsonl_path}')

# Specify the input and output directories
input_dir = 'socialiqa'
output_dir = 'socialiqa_jsonl'
process_directory(input_dir, output_dir)
