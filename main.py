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

    if left_only:
        if left_context_size == 2:
            # todo: calc left tri gram chain
            pass
        else:
            # TODO: calc left bi gram chain
            pass
    elif right_context_size == 2 and left_context_size == 2:
        # TODO: calc left & right & mid tri grams
        pass
    elif left_context_size == 1:
        # TODO: calc left bi gram chain, right & mid tri gram chain
        pass
    else:
        # TODO: calc left & mid tri gram chain, right bi gram chain
        pass

    w_b2, w_b1, w_a1, w_a2 = context


    # todo: implement left_only

    for candidate in candidates:
        # left context
        p_unigram_left = np.log(word2freq[w_b2] + k) - np.log(total_tokens + k * vocab_size)
        left_bigram = " ".join([w_b2, w_b1])
        p_w_b1_give_w_b2 = np.log(word2freq[left_bigram] + k) - np.log(word2freq[w_b2] + k * vocab_size)
        left_trigram = " ".join([left_bigram, candidate]) 
        p_w_b2__w_b1__candidate_given_wb_2__wb_1 = np.log(word2freq[left_trigram] + k) - np.log(
                    word2freq[left_bigram] + k * vocab_size)

        p_left_prob = np.sum(
            np.array([p_unigram_left, p_w_b1_give_w_b2, p_w_b2__w_b1__candidate_given_wb_2__wb_1]))

        # middel context
        mid_p_unigram_left = np.log(word2freq[w_b1] + k) - np.log(total_tokens + k * vocab_size)
        mid_bigram = " ".join([w_b1, candidate])
        p_candidate_given_w_b1 = np.log(word2freq[mid_bigram] + k) - np.log(word2freq[w_b1] + k * vocab_size)
        mid_trigram = " ".join([mid_bigram, w_a1])
        p_wb_1__candidate__w_a1_given_candidate_wb_1 = np.log(word2freq[mid_trigram] + k) - np.log(
                    word2freq[mid_bigram] + k * vocab_size)

        p_mid_prob = np.sum(
            np.array([mid_p_unigram_left, p_candidate_given_w_b1, p_wb_1__candidate__w_a1_given_candidate_wb_1]))

        # right context

        p_right_unigram = np.log(word2freq[candidate] + k) - np.log(total_tokens + k * vocab_size)
        right_bigram = " ".join([candidate, w_a1])
        p_w_a1_given_candidate = np.log(word2freq[w_a1] + k) - np.log(word2freq[candidate] + k * vocab_size)
        right_trigram = " ".join([right_bigram, w_a2])
        p__candidate__w_a1__w_a2_given_candidate_w_a1 = np.log(word2freq[right_trigram] + k) - np.log(
                    word2freq[right_bigram] + k * vocab_size)

        p_right_prob = np.sum(
            np.array([p_right_unigram, p_w_a1_given_candidate, p__candidate__w_a1__w_a2_given_candidate_w_a1]))

        combined_prob = np.sum([p_left_prob, p_mid_prob, p_right_prob])

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
