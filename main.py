import json
import time
from sympy.codegen.ast import continue_
from cloze_utils import *
import numpy as np
from collections import defaultdict
from typing import DefaultDict

import cloze_utils
from debugging import create_word2freq_file, read_word2freq_file

def calc_tri_gram_prob(word2freq, ctx:list, vocab_size, k):
    """Calculate log probability of a trigram with k-smoothing."""
    tri_gram = " ".join(ctx)
    bi_gram = " ".join([ctx[0], ctx[1]])
    numerator = word2freq[tri_gram] + k
    denominator = word2freq[bi_gram] + k * vocab_size
    return numerator / denominator

def predict(word2freq: defaultdict, vocab_size, candidates: list, context: dict,
            k: float, left_only=True) -> str:
    """
    Predict the missing word in a cloze task using an n-gram model with k-smoothing.

    Args:
        word2freq: Dictionary mapping n-grams to their frequencies.
        vocab_size: Size of the vocabulary.
        candidates: List of candidate words to choose from.
        context: Dictionary with 'left_context' and 'right_context' lists.
        k: Smoothing parameter.
        total_tokens: Total number of tokens in the corpus.
        left_only: If True, only use left context for prediction.

    Returns:
        The most likely candidate word to fill the blank.
    """
    max_prob = -np.inf
    best_candidate = None

    for candidate in candidates:
        probs_list = []
        left_context = [context["left_context"][0], context["left_context"][1], candidate]

        if left_only:
            left_tri_gram_prob = calc_tri_gram_prob(word2freq, left_context, vocab_size, k)
            probs_list.append(left_tri_gram_prob)
        else:
            # Use left, middle, and right contexts
            right_context = [candidate, context["right_context"][0], context["right_context"][1]]
            mid_context = [context["left_context"][1], candidate, context["right_context"][0]]

            for ctx in [left_context, mid_context, right_context]:
                ctx_prob = calc_tri_gram_prob(word2freq, ctx, vocab_size, k)
                probs_list.append(ctx_prob)

        # Sum log probabilities
        combined_prob = np.mean(np.array(probs_list))

        if combined_prob > max_prob or best_candidate is None:
            max_prob = combined_prob
            best_candidate = candidate

    return best_candidate

def clean_line(text_line: str) -> str:
    """Cleans an input text input line from punctuation.

    Args:
        text_line (str): The input text line.

    Returns:
        str: text line w/o punctuation.

    """
    text_line = text_line.lower()

    translator = str.maketrans('', '', string.punctuation)
    cleaned_line= text_line.translate(translator)

    return cleaned_line


def train(corpus_filename: str, candidates: set):
    """
        Extract n-gram counts from context windows around candidate words.

        Args:
            corpus_filename: Path to the corpus text file
            candidates: Set of candidate words to build contexts around

        Returns:
            defaultdict mapping n-grams (space-separated strings) to their frequencies
        """
    word2freq = defaultdict(int)
    with open(corpus_filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            cleaned_line = clean_line(line).split()
            padding_before = f"{cloze_utils.SENTENCE_START_SYMBOL} {cloze_utils.SENTENCE_START2_SYMBOL}"
            padding_after = f"{cloze_utils.SENTENCE_END2_SYMBOL} {cloze_utils.SENTENCE_END_SYMBOL}"
            padded_line = [padding_before] + cleaned_line + [padding_after]
            line_len = len(padded_line)

            for i in range(2, line_len - 2):
                word = padded_line[i]
                if word not in candidates:
                    continue

                # left tri gram
                prev_word = padded_line[i - 1]
                prev_prev_word = padded_line[i - 2]
                word2freq[f"{prev_word} {word}"] += 1
                word2freq[f"{prev_prev_word} {prev_word} {word}"] += 1

                # right tri gram
                next_word = padded_line[i + 1]
                next_next_word = padded_line[i + 2]
                word2freq[f"{word} {next_word}"] += 1
                word2freq[f"{word} {next_word} {next_next_word}"] += 1

                # mid trigram
                word2freq[f"{prev_word} {word} {next_word}"] += 1

    return word2freq



def solve_cloze(input_filename, candidates_filename, corpus_filename, left_only):

    predictions = []

    try:
        with open(input_filename, "r", encoding="utf-8") as cloze_file:
            text = cloze_file.read()

    except FileNotFoundError:
        print(f"Error: Could not open file '{cloze_file}' (file not found)")
        return predictions

    try:
        with open(candidates_filename, 'r', encoding='utf-8') as f:
            candidate_list = [line.strip() for line in f]
            candidates = set(candidate_list)
            print(f"Loaded {len(candidates)} candidates.")

    except FileNotFoundError:
        print(f"Error: Candidates file not found at {candidates_filename}")
        return predictions

    print(f'starting to solve the cloze {input_filename} with {candidates} using {corpus_filename}')


    #word2freq = train(corpus_filename,candidates)
    #create_word2freq_file(word2freq, 'word2freq.pkl')

    # To load a pre-trained word2freq dictionary for faster debugging:
    word2freq = read_word2freq_file('word2freq.pkl')
    # print("Loaded word2freq from word2freq.pkl")

    vocab_size = 50000

    contexts_list = get_contexts(input_filename)

    candidate_copy = []

    for l in candidate_list:
        candidate_copy.append(l)
    #print("contexts_list len ", len(contexts_list))
    predictions = []
    for ctx in contexts_list:
        prediction = predict(word2freq, vocab_size, candidate_list, ctx, 0.001)
        predictions.append(prediction)
        candidate_list.remove(prediction)

    true = 0
    for i in range(len(predictions)):
        if predictions[i] == candidate_copy[i]:
            true += 1


    print(f"Accuarcy: {true/len(predictions)*100} | {true} / {len(predictions)}")

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
