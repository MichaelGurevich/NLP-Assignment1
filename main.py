import json
import time
import numpy as np
import string
from collections import defaultdict

from sympy.codegen.ast import continue_

from cloze_utils import *


def calc_unigram_prob(word2freq, w1, tokens_count, vocab_size, k):
    numerator = word2freq[w1] + k
    denominator = tokens_count + k * vocab_size

    return np.log(numerator) - np.log(denominator)

def calc_bi_gram_prob(word2freq, w1, w2, vocab_size, k):
    bi_gram = " ".join([w1, w2])

    numerator = word2freq[bi_gram] + k
    denominator = word2freq[w1] + k * vocab_size

    return np.log(numerator) - np.log(denominator)

def calc_tri_gram_prob(word2freq, w1, w2, w3, vocab_size, k):
    tri_gram = " ".join([w1, w2, w3])
    bi_gram = " ".join([w1, w2])

    numerator = word2freq[tri_gram] + k
    denominator = word2freq[bi_gram] + k * vocab_size

    return np.log(numerator) - np.log(denominator)

def predict(word2freq: defaultdict, vocab_size ,candidates: list, context: dict, k: float, total_tokens, left_only=False) -> str:
    """
    Predicts the missing word in a cloze task using an n-gram model with Add-1 smoothing.

    Args:
        word2freq (defaultdict): A dictionary mapping n-grams to their frequencies.
        candidates (list): A list of candidate words to choose from.
        context (list): A list of four words representing the context around the blank,
                        in the format [word_before_2, word_before_1, word_after_1, word_after_2].

        k (float): k smoothing parameter.

    Returns:
        str: The most likely candidate word to fill the blank.
    """
    # V: Vocabulary size for Add-1 smoothing

    max_prob = -np.inf

    left_context_size = len(context["left_context"])
    right_context_size = len(context["right_context"])
    for candidate in candidates:
        combined_prob = 0
        if left_only:
            if left_context_size == 2: # todo: calc left tri gram chain
                left_context_word_1 = context["left_context"][0]
                left_context_word_2 = context["left_context"][1]
                left_context_word_3 = candidate

                left_unigram_prob = calc_unigram_prob(word2freq, left_context_word_1, total_tokens, vocab_size, k)
                left_bi_gram_prob = calc_bi_gram_prob(word2freq, left_context_word_1, left_context_word_2, vocab_size, k)
                left_tri_gram_prob = calc_tri_gram_prob(word2freq, left_context_word_1, left_context_word_2, left_context_word_3, vocab_size, k)

                combined_prob = np.sum(np.log(np.array([left_unigram_prob, left_bi_gram_prob, left_tri_gram_prob])))
            else: # TODO: calc left bi gram chain
                left_context_word_1 = context["left_context"][0]
                left_context_word_2 = candidate
                left_unigram_prob = calc_unigram_prob(word2freq, left_context_word_1, total_tokens, vocab_size, k)
                left_bi_gram_prob = calc_bi_gram_prob(word2freq, left_context_word_1, left_context_word_2, vocab_size, k)

                combined_prob = np.sum(np.log(np.array([left_unigram_prob, left_bi_gram_prob])))
        elif right_context_size == 2 and left_context_size == 2: # TODO: calc left & right & mid tri grams
            # left context tri gram
            left_context_word_1 = context["left_context"][0]
            left_context_word_2 = context["left_context"][1]
            left_context_word_3 = candidate

            left_unigram_prob = calc_unigram_prob(word2freq, left_context_word_1, total_tokens, vocab_size, k)
            left_bi_gram_prob = calc_bi_gram_prob(word2freq, left_context_word_1, left_context_word_2, vocab_size, k)
            left_tri_gram_prob = calc_tri_gram_prob(word2freq, left_context_word_1, left_context_word_2, left_context_word_3, vocab_size, k)

            left_context_chain_prob = np.sum(np.log(np.array([left_unigram_prob, left_bi_gram_prob, left_tri_gram_prob])))

            # mid context tri gram
            mid_context_word_1 = context["left_context"][1]
            mid_context_word_2 = candidate
            mid_context_word_3 = context["right_context"][0]

            mid_uni_gram_prob = calc_unigram_prob(word2freq, mid_context_word_1, total_tokens, vocab_size, k)
            mid_bi_gram_prob = calc_bi_gram_prob(word2freq, mid_context_word_1, mid_context_word_2, vocab_size, k)
            mi_tri_gram_prob = calc_tri_gram_prob(word2freq, mid_context_word_1, mid_context_word_2, mid_context_word_3, vocab_size, k)

            mid_context_chain_prob = np.sum(np.log(np.array([mid_uni_gram_prob, mid_bi_gram_prob, mi_tri_gram_prob])))

            # right context tri gram
            right_context_word_1 = candidate
            right_context_word_2 = context["right_context"][0]
            right_context_word_3 = context["right_context"][1]

            right_uni_gram_prob = calc_unigram_prob(word2freq, right_context_word_1, total_tokens, vocab_size, k)
            right_bi_gram_prob = calc_bi_gram_prob(word2freq, right_context_word_1, right_context_word_2, candidate, vocab_size, k)
            right_tri_gram_prob = calc_tri_gram_prob(word2freq, right_context_word_1, right_context_word_2, right_context_word_3, vocab_size, k)

            right_context_chain_prob = np.sum(np.log(np.array([right_uni_gram_prob, right_bi_gram_prob, right_tri_gram_prob])))

            combined_prob = np.sum(np.array([left_context_chain_prob, mid_context_chain_prob, right_context_chain_prob]))

        elif left_context_size == 1: # TODO: calc left bi gram chain, right & mid tri gram chain
            # left context bi gram prob
            left_context_word_1 = context["left_context"][0]
            left_context_word_2 = candidate

            left_unigram_prob = calc_unigram_prob(word2freq, left_context_word_1, total_tokens, vocab_size, k)
            left_bi_gram_prob = calc_bi_gram_prob(word2freq, left_context_word_1, left_context_word_2, vocab_size, k)

            left_context_chain_prob = np.sum(np.log(np.array([left_unigram_prob, left_bi_gram_prob])))

            # mid context trigram
            mid_context_word_1 = context["left_context"][1]
            mid_context_word_2 = candidate
            mid_context_word_3 = context["right_context"][0]

            mid_uni_gram_prob = calc_unigram_prob(word2freq, mid_context_word_1, total_tokens, vocab_size, k)
            mid_bi_gram_prob = calc_bi_gram_prob(word2freq, mid_context_word_1, mid_context_word_2, vocab_size, k)
            mi_tri_gram_prob = calc_tri_gram_prob(word2freq, mid_context_word_1, mid_context_word_2, mid_context_word_3, vocab_size, k)

            mid_context_chain_prob = np.sum(np.log(np.array([mid_uni_gram_prob, mid_bi_gram_prob, mi_tri_gram_prob])))

            # right context trigram
            right_context_word_1 = candidate
            right_context_word_2 = context["right_context"][0]
            right_context_word_3 = context["right_context"][1]

            right_uni_gram_prob = calc_unigram_prob(word2freq, right_context_word_1, total_tokens, vocab_size, k)
            right_bi_gram_prob = calc_bi_gram_prob(word2freq, right_context_word_1, right_context_word_2,candidate, vocab_size, k)
            right_tri_gram_prob = calc_tri_gram_prob(word2freq, right_context_word_1, right_context_word_2, right_context_word_3, vocab_size, k)

            right_context_chain_prob = np.sum(np.log(np.array([right_uni_gram_prob, right_bi_gram_prob, right_tri_gram_prob])))

            combined_prob = np.sum(np.array([left_context_chain_prob, mid_context_chain_prob, right_context_chain_prob]))

            pass
        else: # TODO: calc left & mid tri gram chain, right bi gram chain
            # left context tri gram
            left_context_word_1 = context["left_context"][0]
            left_context_word_2 = context["left_context"][1]
            left_context_word_3 = candidate

            left_unigram_prob = calc_unigram_prob(word2freq, left_context_word_1, total_tokens, vocab_size, k)
            left_bi_gram_prob = calc_bi_gram_prob(word2freq, left_context_word_1, left_context_word_2, vocab_size, k)
            left_tri_gram_prob = calc_tri_gram_prob(word2freq, left_context_word_1, left_context_word_2,
                                                    left_context_word_3, vocab_size, k)

            left_context_chain_prob = np.sum(
                np.log(np.array([left_unigram_prob, left_bi_gram_prob, left_tri_gram_prob])))

            # mid context trigram
            mid_context_word_1 = context["left_context"][1]
            mid_context_word_2 = candidate
            mid_context_word_3 = context["right_context"][0]

            mid_uni_gram_prob = calc_unigram_prob(word2freq, mid_context_word_1, total_tokens, vocab_size, k)
            mid_bi_gram_prob = calc_bi_gram_prob(word2freq, mid_context_word_1, mid_context_word_2, vocab_size, k)
            mi_tri_gram_prob = calc_tri_gram_prob(word2freq, mid_context_word_1, mid_context_word_2, mid_context_word_3,
                                                  vocab_size, k)

            mid_context_chain_prob = np.sum(np.log(np.array([mid_uni_gram_prob, mid_bi_gram_prob, mi_tri_gram_prob])))

            # right context bi gram
            right_context_word_1 = candidate
            right_context_word_2 = context["right_context"][0]

            right_uni_gram_prob = calc_unigram_prob(word2freq, right_context_word_1, total_tokens, vocab_size, k)
            right_bi_gram_prob = calc_bi_gram_prob(word2freq, right_context_word_1, right_context_word_2, candidate,
                                                   vocab_size, k)

            right_context_chain_prob = np.sum(np.log(np.array([right_uni_gram_prob, right_bi_gram_prob,])))

            combined_prob = np.sum(np.array([left_context_chain_prob, mid_context_chain_prob, right_context_chain_prob]))


        if combined_prob > max_prob:
            max_prob = combined_prob
            best_candidate = candidate

    return best_candidate


def train(corpus_filename: str, candidates_list: list) -> dict:
    word2freq = defaultdict(int)
    candidates_set = set(candidates_list)

    print(f"Loaded {len(candidates_list)} candidates.")

    unigram_count = 0
    bigram_count = 0
    trigram_count = 0

    try:
        with open(corpus_filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                cleaned_line = clean_line(line).split()
                line_len = len(cleaned_line)

                for i, word in enumerate(cleaned_line):
                    if word not in candidates_set:
                        continue
                # todo: change edge cases of candidate appearing at the start/2nd/3rd/last/second to last place due to adding <s>
                    # Left bigram
                    if i > 0:
                        ngram = cleaned_line[i - 1:i + 1]
                        word2freq[" ".join(ngram)] += 1
                        bigram_count += 1
                        for w in ngram:
                            word2freq[w] += 1
                            unigram_count += 1

                    # Right bigram
                    if i < line_len - 1:
                        ngram = cleaned_line[i:i + 2]
                        word2freq[" ".join(ngram)] += 1
                        bigram_count += 1
                        for w in ngram:
                            word2freq[w] += 1
                            unigram_count += 1

                    # Left trigram
                    if i > 1:
                        ngram = cleaned_line[i - 2:i + 1]
                        word2freq[" ".join(ngram)] += 1
                        trigram_count += 1
                        for w in ngram:
                            word2freq[w] += 1
                            unigram_count += 1

                    # Middle trigram
                    if i > 0 and i < line_len - 1:
                        ngram = cleaned_line[i - 1:i + 2]
                        word2freq[" ".join(ngram)] += 1
                        trigram_count += 1
                        for w in ngram:
                            word2freq[w] += 1
                            unigram_count += 1

                    # Right trigram
                    if i < line_len - 2:
                        ngram = cleaned_line[i:i + 3]
                        word2freq[" ".join(ngram)] += 1
                        trigram_count += 1
                        for w in ngram:
                            word2freq[w] += 1
                            unigram_count += 1

    except FileNotFoundError:
        print(f"Error: Corpus file not found at {corpus_filename}")
        return word2freq

    save_probabilities_to_file(word2freq, "ngram_dict.pkl")
    print("saved probabilities to ngram_dict.pkl")

    print(unigram_count, bigram_count, trigram_count)
    return word2freq



def solve_cloze(input, candidates, corpus, left_only):
    # todo: implement this function

    print(f'starting to solve the cloze {input} with {candidates} using {corpus}')

    with open(candidates, 'r', encoding='utf-8') as f:
        candidate_list = [line.strip() for line in f]

    word2freq = train(corpus, candidate_list)

    return list()


if __name__ == '__main__':
    start_time = time.time()

    with open('config.json', 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    solution = solve_cloze(config['input_filename'],
                           config['candidates_filename'],
                           config['corpus'],
                           config['left_only'])

    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time:.2f} seconds")

    print('cloze solution:', solution)
