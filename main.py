import json
import time
import random
import numpy as np

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


    


def solve_cloze(input, candidates, corpus, left_only):
    # todo: implement this function
    print(f'starting to solve the cloze {input} with {candidates} using {corpus}')

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
