import re
import string
import numpy as np
import random
from dask.dataframe.methods import value_counts_aggregate
from distributed.utils_comm import retry

#function that find the indices of the blanks in a string
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
# returns a list of strings [previous, next] or [previous] or an empty list (if left = true) of the n words beside blank.

def get_context(text, blank_indices, n=1, left = False):

    start, end = blank_indices

    #splitting the string into two parts
    before = text [:start]
    after = text [end+1:]

    #splitting each part to a list of words
    #NOTICE: i allowed hyphens and apostrophes in words so well-known and tim's would be one word,
    #but we can change that depending on how will build the module
    before_words = re.findall(r"[\w'-]+", before)
    after_words = re.findall(r"[\w'-]+", after)


    #checking if there are enough words on each side of a blank,
    #but assigns what's there anyway

    prev_n = before_words[-n:] if len(before_words) >= n else before_words
    next_n = after_words[:n] if len(after_words) >= n else after_words



    #concatenating the lists to strings
    #NOTICE: this will return a list of strings even if there's less than we are expecting
    if left:
        return [" ".join(prev_n)]
    else:
        return [" ".join(prev_n), " ".join(next_n)]


def clean_line(text_line: str) -> str:
    """Cleans an input text input line from punctuation.

    Args:
        text_line (str): The input text line.

    Returns:
        str: text line w/o punctuation.

    """
    text_line = text_line.lower()

    translator = str.maketrans('', '', string.punctuation + "“”")
    cleaned_line= text_line.translate(translator)

    return cleaned_line


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


import pickle

def save_probabilities_to_file(probabilities_dict, file_path):
    """
    Saves a dictionary of probabilities to a file using pickle.

    Args:
        probabilities_dict (dict): The dictionary of probabilities to save.
        file_path (str): The path to the file where the dictionary will be saved.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(probabilities_dict, f)

def load_probabilities_from_file(file_path):
    """
    Loads a dictionary of probabilities from a file using pickle.

    Args:
        file_path (str): The path to the file from which to load the dictionary.

    Returns:
        dict: The loaded dictionary of probabilities.
    """
    with open(file_path, 'rb') as f:
        probabilities_dict = pickle.load(f)
    return probabilities_dict