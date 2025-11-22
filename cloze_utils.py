
import re

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