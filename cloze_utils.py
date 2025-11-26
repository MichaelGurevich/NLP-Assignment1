"""Utility functions and constants for the cloze-solving task.

This module provides helper functions for text cleaning, sentence padding, and
extracting contexts around blanks in a cloze puzzle. It also defines several
constants used throughout the project.
"""
import string

SENTENCE_START_SYMBOL = "<s>"
SENTENCE_END_SYMBOL = "</s>"
SENTENCE_START2_SYMBOL = "<s2>"
SENTENCE_END2_SYMBOL = "</s2>"
CLOZE_BLANK_SYMBOL = "__________"
VOCAB_SIZE = 50000
K_SMOOTHING_VALUE = 0.0005


def clean_line(text_line: str) -> str:
    """Cleans a line of text by lowercasing and removing punctuation.

    Args:
        text_line (str): The raw input text line.

    Returns:
        str: The cleaned text line.
    """
    text_line = text_line.lower()
    translator = str.maketrans('', '', string.punctuation)
    cleaned_line = text_line.translate(translator)
    return cleaned_line


def pad_line(cleaned_line: list) -> list:
    """Adds padding symbols to the start and end of a tokenized sentence.

    The padding consists of two symbols at the start and two at the end to ensure
    that a trigram model has a valid two-word context for every word in the
    original sentence.

    Args:
        cleaned_line (list[str]): A list of tokens representing a sentence.

    Returns:
        list[str]: The list of tokens with padding symbols added.
    """
    return [SENTENCE_START_SYMBOL, SENTENCE_START2_SYMBOL] + cleaned_line + [SENTENCE_END2_SYMBOL, SENTENCE_END_SYMBOL]


def get_contexts(cloze_text: str) -> list[dict]:
    """Extracts the contexts surrounding each blank in a cloze text.

    This function processes a block of text containing one or more cloze blanks
    (represented by CLOZE_BLANK_SYMBOL). For each blank, it extracts the two
    words immediately to the left and the two words immediately to the right.

    Args:
        cloze_text (str): The full string of the cloze puzzle.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a
                    blank. The dictionary has two keys: "left_context" and
                    "right_context", each containing a list of two words.
    """
    # Create a translator that removes all punctuation except underscores,
    # to protect the CLOZE_BLANK_SYMBOL.
    punctuation_without_underscore = string.punctuation.replace('_', '')
    translator = str.maketrans('', '', punctuation_without_underscore)

    contexts_list = []

    for line in cloze_text.splitlines():
        if not line.strip():
            continue

        cleaned_line = line.translate(translator).split()
        padded_line = pad_line(cleaned_line)

        for i in range(2, len(padded_line) - 2):
            if padded_line[i] != CLOZE_BLANK_SYMBOL:
                continue

            context = {
                "left_context": [padded_line[i - 2], padded_line[i - 1]],
                "right_context": [padded_line[i + 1], padded_line[i + 2]]
            }
            contexts_list.append(context)

    return contexts_list