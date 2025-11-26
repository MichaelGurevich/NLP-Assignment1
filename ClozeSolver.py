"""Contains the core logic for the trigram-based cloze solving model."""
from collections import defaultdict
from cloze_utils import *
import numpy as np


class ClozeSolver:
    """A trigram language model for solving cloze puzzles.

    This class implements a simplified trigram model focused on a specific task:
    solving cloze puzzles where the candidate words are known. It uses k-smoothing
    for probability calculations.

    The training process is optimized to only count n-grams that occur around a
    pre-defined set of candidate words, making it memory-efficient for large
    corpora.

    Attributes:
        k (float): The smoothing factor for k-smoothing.
        vocab_size (int): The size of the vocabulary, used in the denominator
            for k-smoothing.
        word2freq (defaultdict): A dictionary mapping n-grams (as space-separated
            strings) to their frequency counts in the corpus.
    """

    def __init__(self, k: float, vocab_size: int):
        """Initializes the ClozeSolver.

        Args:
            k (float): The smoothing factor (e.g., 1.0 for add-one smoothing).
            vocab_size (int): The total size of the vocabulary for smoothing.
        """
        self.k = k
        self.vocab_size = vocab_size
        self.word2freq = defaultdict(int)
        # NOTE(intern): This attribute is initialized but never used.
        # It's good practice to remove unused variables to keep the code clean.
        self.total_tokens = 0

    @staticmethod
    def calc_tri_gram_prob(word2freq, ctx: list, vocab_size, k):
        """Calculates the k-smoothed probability of a trigram.

        Args:
            word2freq (defaultdict): A dictionary of n-gram counts.
            ctx (list[str]): A list of three words representing the trigram.
            vocab_size (int): The size of the vocabulary for smoothing.
            k (float): The smoothing factor.

        Returns:
            float: The smoothed probability of the trigram.
        """
        tri_gram = " ".join(ctx)
        bi_gram = " ".join([ctx[0], ctx[1]])
        numerator = word2freq[tri_gram] + k

        denominator = word2freq[bi_gram] + k * vocab_size
        return numerator / denominator

    def fit(self, corpus_filename: str, candidates: set):
        """Trains the model by counting relevant n-grams from a corpus.

        This method builds the `word2freq` frequency map. It iterates through the
        corpus and, for each line, finds occurrences of the candidate words.
        It then counts the trigrams and bigrams that form a window around each
        found candidate. This is a memory-saving optimization.

        Args:
            corpus_filename (str): Path to the corpus text file.
            candidates (set[str]): A set of candidate words to build contexts around.
        """
        word2freq = defaultdict(int)
        with open(corpus_filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                cleaned_line = clean_line(line).split()
                padded_line = pad_line(cleaned_line)
                line_len = len(padded_line)

                for i in range(2, line_len - 2):
                    word = padded_line[i]
                    if word not in candidates:
                        continue

                    prev_word = padded_line[i - 1]
                    prev_prev_word = padded_line[i - 2]
                    next_word = padded_line[i + 1]
                    next_next_word = padded_line[i + 2]

                    # left context bi and tri grams
                    word2freq[f"{prev_prev_word} {prev_word}"] += 1
                    word2freq[f"{prev_prev_word} {prev_word} {word}"] += 1

                    # right context bi and tri grams
                    word2freq[f"{word} {next_word}"] += 1
                    word2freq[f"{word} {next_word} {next_next_word}"] += 1

                    # mid context bi and tri grams
                    word2freq[f"{prev_word} {word}"] += 1
                    word2freq[f"{prev_word} {word} {next_word}"] += 1

        self.word2freq = word2freq

    def predict(self, candidates: list, context: dict, left_only: bool = False) -> str:
        """Predicts the best word for a blank based on its context.

        For each candidate word, this method calculates a score based on the
        probability of the trigrams that would be formed if the candidate were
        placed in the blank. It considers the left, right, and middle trigram
        contexts, unless `left_only` is True.

        Args:
            candidates (list[str]): A list of candidate words to choose from.
            context (dict): A dictionary with 'left_context' (list of 2 words)
                and 'right_context' (list of 2 words).
            left_only (bool): If True, only the left context (bigram) is used
                for prediction.

        Returns:
            str: The candidate word with the highest probability score.
        """
        max_prob = -np.inf
        best_candidate = None

        for candidate in candidates:
            probs_list = []
            left_context = [context["left_context"][0], context["left_context"][1], candidate]

            if left_only:
                left_tri_gram_prob = self.calc_tri_gram_prob(self.word2freq, left_context, self.vocab_size, self.k)
                probs_list.append(left_tri_gram_prob)
            else:
                right_context = [candidate, context["right_context"][0], context["right_context"][1]]
                mid_context = [context["left_context"][1], candidate, context["right_context"][0]]

                for ctx in [left_context, mid_context, right_context]:
                    ctx_prob = self.calc_tri_gram_prob(self.word2freq, ctx, self.vocab_size, self.k)
                    probs_list.append(ctx_prob)

            combined_prob = np.sum(np.log(np.array(probs_list)))

            if combined_prob > max_prob or best_candidate is None:
                max_prob = combined_prob
                best_candidate = candidate

        return best_candidate