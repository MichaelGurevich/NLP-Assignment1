"""Solves a cloze puzzle by training a trigram model on a corpus.

This script serves as the main entry point for the cloze-solving task. It reads
a configuration file, trains a language model based on the specified corpus,
and then predicts the missing words in a cloze text from a given list of
candidates.
"""
import json
import time
from cloze_utils import *
from ClozeSolver import ClozeSolver


def solve_cloze(input_filename, candidates_filename, corpus_filename, left_only):
    """Solves a cloze puzzle using a trigram language model.

    This function orchestrates the entire cloze-solving process. It handles
    file I/O, initializes the ClozeSolver, triggers the training ('fit') phase,
    and then iteratively predicts the missing words.

    Args:
        input_filename (str): Path to the text file containing the cloze puzzle.
        candidates_filename (str): Path to the text file containing the
            candidate words, one per line.
        corpus_filename (str): Path to the large text corpus for training the
            language model.
        left_only (bool): If True, the model uses only the left context (bigram)
            for prediction. If False, it uses the full trigram context.

    Returns:
        list[str]: A list of predicted words in the order they should appear
                   in the cloze text. Returns an empty list if file errors occur.
    """
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

    solver = ClozeSolver(k=K_SMOOTHING_VALUE, vocab_size=VOCAB_SIZE)

    # Train the model
    solver.fit(corpus_filename, candidates)

    # Get contexts from the cloze file
    contexts_list = get_contexts(text)

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