import string


SENTENCE_START_SYMBOL = "<s>"
SENTENCE_END_SYMBOL = "</s>"
SENTENCE_START2_SYMBOL = "<s2>"
SENTENCE_END2_SYMBOL = "</s2>"
CLOZE_BLANK_SYMBOL = "__________"
VOCAB_SIZE = 50000
K_SMOOTHING_VALUE = 0.0005


def clean_line(text_line: str) -> str:
    """Cleans an input text input line from punctuation.

    Args:
        text_line (str): The input text line.

    Returns:
        str: text line w/o punctuation.

    """
    text_line = text_line.lower()

    translator = str.maketrans('', '', string.punctuation)
    cleaned_line = text_line.translate(translator)

    return cleaned_line


def pad_line(cleaned_line: list) -> list:
    """Add sentence start/end padding to a cleaned line.

    Args:
        cleaned_line: List of tokens (already cleaned/split)

    Returns:
        List with sentence start and end symbols added
    """
    return [SENTENCE_START_SYMBOL, SENTENCE_START2_SYMBOL] + cleaned_line + [SENTENCE_END2_SYMBOL, SENTENCE_END_SYMBOL]


def get_contexts(cloze_text: str) -> list:
    """
     Extract left and right context words (up to 2 on each side) surrounding cloze blanks.

     Args:
         cloze_text: Content of file with sentences containing cloze blanks

     Returns:
         List of dicts with "left_context" and "right_context" for each blank

     Note: Assumes cloze blanks are not the only word in a sentence.
     """

    punctuation_without_underscore = string.punctuation.replace('_', '')
    translator = str.maketrans('', '', punctuation_without_underscore)

    contexts_list = []

    for line in cloze_text.splitlines():
        cleaned_line = line.translate(translator).split()
        # add start and end sentence symbols
        padded_line = pad_line(cleaned_line)

        for i in range(2, len(padded_line) - 2):
            if padded_line[i] != CLOZE_BLANK_SYMBOL:
                continue

            context = {
                "left_context": [],
                "right_context": []
            }

            prev_word = padded_line[i - 1]
            prev_prev_word = padded_line[i - 2]
            next_word = padded_line[i + 1]
            next_next_word = padded_line[i + 2]

            context["left_context"] = [prev_prev_word, prev_word]
            context["right_context"] = [next_word, next_next_word]

            contexts_list.append(context)

    return contexts_list