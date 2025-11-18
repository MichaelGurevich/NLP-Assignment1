import json
import time


def get_solution_accuracy(label: list, solution: list) -> float:
    list_len = len(label)

    if list_len == 0:
        return 1.0

    if list_len != len(solution):
        return 0
    
    predicted_correct = 0
    for i in range(list_len):
        if label[i] == solution[i]:
            predicted_correct += 1
    
    return predicted_correct / list_len




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
