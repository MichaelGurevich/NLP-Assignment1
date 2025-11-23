import string
import re

SENTENCE_START_SYMBOL = "<s>"
SENTENCE_END_SYMBOL = "</s>"
CLOZE_BLANK_SYMBOL = "__________"

#function that finds the indices of the blanks in a string
# returns a list of tuples (starting index, end index).
def find_blanks(text):
    blanks = []
    in_Blank = False
    start = None

    for index, char in enumerate(text):
        if char == '_':
            if in_Blank == False:  #found start of new blank
                start = index
                in_Blank = True
        else:
            if in_Blank:  # found end of blank
                blanks.append((start, index-1))
                in_Blank = False


    if in_Blank: #end of string is blank
        blanks.append((start, len(text)-1))

    return blanks

# function that receives a tuple of blank indices in string,
# n - number of words to retrieve on either side of blank,
# left - flag for whether to return the words only to the left of blank or from both sides
# returns a list of words [before1, before2, after1, after2] or [before1, before2] (if left = true) of the n words beside blank.

def get_context(text, blank_indices, n=1, left = False):

    start, end = blank_indices

    #splitting the string into two parts
    before = text [:start]
    after = text [end+1:]

    #splitting each part to a list of words
    before_words = re.findall(r"\w+", before)
    after_words = re.findall(r"\w+", after)


    #checking if there are enough words on each side of a blank,
    #but assigns what's there anyway
    prev_n = before_words[-n:] if len(before_words) >= n else before_words
    next_n = after_words[:n] if len(after_words) >= n else after_words


    #NOTICE: this will return a list of strings even if there's less than we are expecting
    if left:
        return prev_n
    else:
        return prev_n + next_n


def get_all_contexts(text, n=2, left=False):
    blanks = find_blanks(text)
    return [get_context(text, b, n=n, left=left) for b in blanks]



def get_contexts(cloze_filename:str) -> list:
    """
     Extract left and right context words (up to 2 on each side) surrounding cloze blanks.

     Args:
         cloze_filename: Path to file with sentences containing cloze blanks

     Returns:
         List of dicts with "left_context" and "right_context" for each blank

     Note: Assumes cloze blanks are not the only word in a sentence.
     """

    punctuation_without_underscore = string.punctuation.replace('_', '')
    translator = str.maketrans('', '', punctuation_without_underscore)

    contexts_list = []


    with open(cloze_filename, 'r', encoding='utf-8') as fin:
        for line in fin:

            cleaned_line = line.translate(translator).split()

            # add start and end sentence symbols
            cleaned_line.insert(0, SENTENCE_START_SYMBOL)
            cleaned_line.append(SENTENCE_END_SYMBOL)

            for i in range (1, len(cleaned_line)-1, 1):
                if cleaned_line[i] != CLOZE_BLANK_SYMBOL:
                    continue

                context = {
                    "left_context": [],
                    "right_context": []
                }

                prev_word = cleaned_line[i-1]
                next_word = cleaned_line[i+1]

                if prev_word == SENTENCE_START_SYMBOL: # candidate is the 1st word in the sentence
                    context["left_context"].append(SENTENCE_START_SYMBOL)
                    context["right_context"].append(next_word)
                    context["right_context"].append(cleaned_line[i+2])
                elif next_word == SENTENCE_END_SYMBOL:
                    context["right_context"].append(SENTENCE_END_SYMBOL)
                    context["left_context"].append(cleaned_line[i - 2])
                    context["left_context"].append(prev_word)
                else:
                    context["right_context"].append(next_word)
                    context["right_context"].append(cleaned_line[i+2])

                    context["left_context"].append(cleaned_line[i - 2])
                    context["left_context"].append(prev_word)


                contexts_list.append(context)

    return contexts_list