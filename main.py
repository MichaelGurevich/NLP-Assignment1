import json
import time
from cloze_utils import *
from ClozeSolver import ClozeSolver


def solve_cloze(input_filename, candidates_filename, corpus_filename, left_only):

    predictions = []

    try:
        with open(input_filename, "r", encoding="utf-8") as cloze_file:
            text = cloze_file.read()

    except FileNotFoundError:
        print(f"Error: Could not open file '{input_filename}' (file not found)")
        return predictions

    try:
        with open(candidates_filename, 'r', encoding='utf-8') as f:
            candidate_list = [line.strip() for line in f]
            candidates = set(candidate_list)
            print(f"Loaded {len(candidates)} candidates.")

    except FileNotFoundError:
        print(f"Error: Candidates file not found at {candidates_filename}")
        return predictions

    print(f'starting to solve the cloze {input_filename} with {len(candidates)} candidates using {corpus_filename}')

    # Initialize the ClozeSolver
    solver = ClozeSolver(k=K_SMOOTHING_VALUE, vocab_size=VOCAB_SIZE)

    # Train the model
    solver.fit(corpus_filename, candidates)

    # Get contexts from the cloze file
    contexts_list = get_contexts(input_filename)

    # Make predictions
    predictions = []
    for ctx in contexts_list:
        prediction = solver.predict(candidate_list, ctx, left_only=left_only)
        predictions.append(prediction)
        candidate_list.remove(prediction)

    return predictions


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    solution = solve_cloze(config['input_filename'],
                           config['candidates_filename'],
                           config['corpus_filename'],
                           config['left_only'])

    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time:.2f} seconds")

    print('cloze solution:', solution)