import json
import time
import random
import numpy as np
import string
from collections import defaultdict

from distributed.utils_comm import retry


def get_solution_accuracy(label: list, solution: list) -> float:
    """Calculates the accuracy of a given solution against a label.

    Args:
        label (list): The ground truth labels.
        solution (list): The predicted solution.

    Returns:
        float: The accuracy of the solution, a float between 0.0 and 1.0.
    """
    label_array = np.array(label)
    solution_array = np.array(solution)

    list_len = len(label_array)

    if list_len == 0:
        return 1.0

    if list_len != len(solution_array):
        return 0.0

    predicted_correct = np.sum(label_array == solution_array)
    
    return predicted_correct / list_len


def get_random_solution_baseline(num_of_solutions: int, cloze_word_list: list) -> float:
    """Calculates the average accuracy of multiple random cloze solutions as a baseline.

    This function combines the generation of random solutions and the calculation
    of their average accuracy.

    Args:
        num_of_solutions (int): The number of random solutions to generate.
        cloze_word_list (list): The list of words representing the true solution (label).

    Returns:
        float: The average accuracy across all generated random solutions.
    """
    cloze_word_array = np.array(cloze_word_list)
    label_array = cloze_word_array

    accuracy_list = []
    for _ in range(num_of_solutions):
        # Generate a single random solution as a numpy array
        random_solution_array = np.random.permutation(cloze_word_array)
        
        # Calculate accuracy for this random solution
        accuracy = get_solution_accuracy(label_array, random_solution_array)
        accuracy_list.append(accuracy)
    
    return np.mean(accuracy_list)


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


def predict(word2freq: defaultdict, vocab_size ,candidates: list, context: list, k: float) -> str:
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

    if vocab_size == 0:
        # Avoid division by zero if vocab is empty
        return candidates[0] if candidates else ""

    max_prob = -1.0
    best_candidate = candidates[0] if candidates else ""

    """
    if len(context) != 4:
        print("Warning: Context must be a list of 4 words. Using a dummy context.")
        context = ["", "", "", ""]
    """

    w_b2, w_b1, w_a1, w_a2 = context

    # todo: implement fall back if trigram not found try using bi-gram
    # todo: implement left_only

    for candidate in candidates:
        """
        # P(candidate | w_b1) with Add-1 Smoothing
        left_bigram_hist = w_b1
        left_bigram = f"{w_b1} {candidate}"
        left_bigram_prob = (word2freq.get(left_bigram, 0) + k) / (word2freq.get(left_bigram_hist, 0) + k * vocab_size)
        """

        # P(candidate | w_b2, w_b1) with Add-1 Smoothing
        left_trigram_hist = f"{w_b2} {w_b1}"
        left_trigram = f"{w_b2} {w_b1} {candidate}"
        left_trigram_prob = (word2freq.get(left_trigram, 0) + k) / (word2freq.get(left_trigram_hist, 0) + k * vocab_size)

        """
        # P(w_a1 | candidate) with Add-1 Smoothing
        right_bigram_hist = candidate
        right_bigram = f"{candidate} {w_a1}"
        right_bigram_prob = (word2freq.get(right_bigram, 0) + k) / (word2freq.get(right_bigram_hist, 0) + k * vocab_size)
        """

        # P(w_a2 | candidate, w_a1) with Add-1 Smoothing
        right_trigram_hist = f"{candidate} {w_a1}"
        right_trigram = f"{candidate} {w_a1} {w_a2}"
        right_trigram_prob = (word2freq.get(right_trigram, 0) + k) / (word2freq.get(right_trigram_hist, 0) + k * vocab_size)


        # P(w_a1 | w_b1, candidate) with Add-1 Smoothing
        mid_trigram_hist = f"{w_b1} {candidate}"
        mid_trigram = f"{w_b1} {candidate} {w_a1}"
        mid_trigram_prob = (word2freq.get(mid_trigram, 0) + k) / (word2freq.get(mid_trigram_hist, 0) + k * vocab_size)



        # The original code took the max of the probabilities. We preserve this approach.
        #prob_arr = np.array([left_bigram_prob, left_trigram_prob, right_bigram_prob, right_trigram_prob, mid_trigram_prob])


        prob_arr = np.array([left_trigram_prob, right_trigram_prob, mid_trigram_prob])
        curr_max_prob = np.max(prob_arr)

        if curr_max_prob > max_prob:
            max_prob = curr_max_prob
            best_candidate = candidate

    return best_candidate




def train(corpus_filename: str, candidates_filename: str) -> dict:
    word2freq = defaultdict(int)

    try:
        with open(candidates_filename, 'r', encoding='utf-8') as f:
            candidate_list = [line.strip() for line in f]
            candidates = set(candidate_list)

            # todo: add at the start of the each line <s>
            # todo: add at the end of the each line <\s>
    except FileNotFoundError:
        print(f"Error: Candidates file not found at {candidates_filename}")
        return word2freq

    print(f"Loaded {len(candidates)} candidates.")

    unigram_count = 0
    bigram_count = 0
    trigram_count = 0

    try:
        with open(corpus_filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                cleaned_line = clean_line(line).split()
                line_len = len(cleaned_line)

                for i, word in enumerate(cleaned_line):
                    if word not in candidates:
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

    print(unigram_count, bigram_count, trigram_count)
    # Calling predict with a dummy context, as in the original implementation.

    contexts = [
        ["included", "positional", "were", "not"],
        ["the", "vigesimal", "that", "defined"],
        ["the", "isolated", "area", "of"],
        ["a", "more", "area", "due"],
        ["to", "human", "enabling", "more"],
        ["convenient", "and", "transport", "predominantly"],
        ["the", "more", "used", "fonts"],
        ["processors", "web", "and", "other"],
        ["unrecognizable", "when", "in", "a"],
        ["40", "million", "make", "their"],
        ["at", "least", "of", "which"],
        ["better", "diagnostic", "for", "tuberculosis"],
    ]
    i = 1
    vocab_size = len({k for k in word2freq.keys() if len(k.split()) == 1})
    predictions = []
    for context in contexts:
        prediction = predict(word2freq, vocab_size, candidate_list, context, 0.02)
        predictions.append(prediction)

    print(predictions)
    """
    for k in np.arange(0.000001, 0.04, 0.000001):
        predictions = []
        for context in contexts:
            prediction = predict(word2freq,vocab_size ,candidate_list, context, k)
            predictions.append(prediction)

        true = 0
        for label, prediction in zip(candidate_list, predictions):
            if label == prediction:
                true += 1

        i += 1

        acc = true / len(candidate_list)
        if acc == 1.0:
            print(f"i: {i} | k: {k} | accuracy {acc}% +++++++++++++++++++++++++++++++++")
        else:
            print(f"i: {i} | k: {k} | accuracy {acc}% ")
    """
    return word2freq



def solve_cloze(input, candidates, corpus, left_only):
    # todo: implement this function

    print(f'starting to solve the cloze {input} with {candidates} using {corpus}')

    train(corpus, candidates)

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
