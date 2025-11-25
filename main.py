import json
import time
from sympy.codegen.ast import continue_
from cloze_utils import *
import numpy as np
from collections import defaultdict
from typing import DefaultDict

import cloze_utils
from debugging import create_word2freq_file, read_word2freq_file


def calc_unigram_prob(word2freq, w1, tokens_count, vocab_size, k):
    """Calculate log probability of a unigram with k-smoothing."""
    numerator = word2freq[w1] + k
    denominator = tokens_count + k * vocab_size
    return np.log(numerator / denominator)


def calc_bi_gram_prob(word2freq, w1, w2, vocab_size, k):
    """Calculate log probability of a bigram with k-smoothing."""
    bi_gram = " ".join([w1, w2])
    numerator = word2freq[bi_gram] + k
    denominator = word2freq[w1] + k * vocab_size
    return np.log(numerator / denominator)


def calc_tri_gram_prob(word2freq, w1, w2, w3, vocab_size, k):
    """Calculate log probability of a trigram with k-smoothing."""
    tri_gram = " ".join([w1, w2, w3])
    bi_gram = " ".join([w1, w2])
    numerator = word2freq[tri_gram] + k
    denominator = word2freq[bi_gram] + k * vocab_size
    return np.log(numerator / denominator)


def calc_chain_prob(word2freq, words: list, vocab_size, total_tokens, k):
    """
    Calculate total log probability of a word sequence (2 or 3 words).
    Combines unigram, bigram, and optionally trigram probabilities.
    """
    uni_gram_prob = calc_unigram_prob(word2freq, words[0], total_tokens, vocab_size, k)
    bi_gram_prob = calc_bi_gram_prob(word2freq, words[0], words[1], vocab_size, k)

    if len(words) == 3:
        tri_gram_prob = calc_tri_gram_prob(word2freq, words[0], words[1], words[2], vocab_size, k)
        prob_arr = np.array([uni_gram_prob, bi_gram_prob, tri_gram_prob])
    else:
        prob_arr = np.array([uni_gram_prob, bi_gram_prob])

    return sum(prob_arr)


def get_right_context(context: list, context_size: int, candidate):
    """Build right context sequence: [candidate, word_after_1, word_after_2?]"""
    return [candidate, context[0], context[1]] if context_size == 2 else [candidate, context[0]]


def get_left_context(context: list, context_size: int, candidate):
    """Build left context sequence: [word_before_2?, word_before_1, candidate]"""
    return [context[0], context[1], candidate] if context_size == 2 else [context[0], candidate]


def predict(word2freq: defaultdict, vocab_size, candidates: list, context: dict,
            k: float, total_tokens, left_only=False) -> str:
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

    left_context_size = len(context["left_context"])
    right_context_size = len(context["right_context"])

    for candidate in candidates:
        probs_list = []
        left_context = get_left_context(context["left_context"], left_context_size, candidate)

        if left_only:
            # Use only left context for prediction
            left_only_context_prob = calc_chain_prob(word2freq, left_context, vocab_size, total_tokens, k)
            probs_list.append(left_only_context_prob)
        else:
            # Use left, middle, and right contexts
            right_context = get_right_context(context["right_context"], right_context_size, candidate)
            mid_context = [context["left_context"][-1], candidate, context["right_context"][0]]

            for ctx in [left_context, mid_context, right_context]:
                ctx_prob = calc_chain_prob(word2freq, ctx, vocab_size, total_tokens, k)
                probs_list.append(ctx_prob)

        # Sum log probabilities
        combined_prob = np.sum(np.array(probs_list))

        if combined_prob > max_prob:
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
    unigram_count = 0
    with open(corpus_filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            cleaned_line = clean_line(line).split()
            padded_line = [cloze_utils.SENTENCE_START_SYMBOL] + cleaned_line + [cloze_utils.SENTENCE_END_SYMBOL]
            line_len = len(padded_line)

            for i in range(1, line_len - 1):
                word = padded_line[i]
                if word not in candidates:
                    continue

                prev_word = padded_line[i - 1]
                next_word = padded_line[i + 1]

                # Count unigrams in the window
                word2freq[prev_word] += 1
                word2freq[word] += 1
                word2freq[next_word] += 1
                unigram_count += 3

                # Count bigrams
                word2freq[f"{prev_word} {word}"] += 1
                word2freq[f"{word} {next_word}"] += 1

                # Left trigram if available
                if i - 2 >= 0:
                    prev_prev_word = padded_line[i - 2]
                    word2freq[prev_prev_word] += 1
                    word2freq[f"{prev_prev_word} {prev_word} {word}"] += 1

                # Right trigram if available
                if i + 2 < line_len:
                    next_next_word = padded_line[i + 2]
                    word2freq[next_next_word] += 1
                    word2freq[f"{word} {next_word} {next_next_word}"] += 1

                # Middle trigram
                if prev_word != cloze_utils.SENTENCE_START_SYMBOL and next_word != cloze_utils.SENTENCE_END_SYMBOL:
                    word2freq[f"{prev_word} {word} {next_word}"] += 1

    return word2freq, unigram_count



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


    word2freq, unigram_count = train(corpus_filename,candidates)
    #create_word2freq_file(word2freq, 'word2freq.pkl')

    # To load a pre-trained word2freq dictionary for faster debugging:
    #word2freq = read_word2freq_file('word2freq.pkl')
    # print("Loaded word2freq from word2freq.pkl")

    vocab_size = len({k for k in word2freq if ' ' not in k})

    #vocab_size = 138502
    #unigram_count = 2554389

    contexts_list = get_contexts(input_filename)



    #print("contexts_list len ", len(contexts_list))
    predictions = []
    for ctx in contexts_list:
        print(ctx)
        prediction = predict(word2freq, vocab_size, candidate_list, ctx, 0.01, unigram_count)
        predictions.append(prediction)
        candidate_list.remove(prediction)


    for pred in predictions:
        print(pred)
    """
    vocab_size = len({k for k in word2freq if ' ' not in k})
    for context in contexts:
        prediction = predict(word2freq, vocab_size, candidate_list, context, 0.02)
        predictions.append(prediction)

    """
    return []


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
