"""
Create and write a program inspection.py to calculate the label entropy at the root
(i.e. the entropy of the labels before any splits) and the error rate (the percent of incorrectly
classified instances) of classifying using a majority vote (picking the label with the most
examples). You do not need to look at the values of any of the attributes to do these calculations;
knowing the labels of each example is sufficient. Entropy should be calculated in bits using log
base 2
"""

import sys
import numpy as np # type: ignore


def load_file_contents(cmd_argument_number: int):
    input_file_name = sys.argv[cmd_argument_number]
    print(f"Loading input file: {input_file_name}")
    input_data = np.genfromtxt(
        fname=input_file_name, delimiter="\t", dtype=None, encoding=None
    )
    input_data_inputs = input_data[1:, 0 : input_data.shape[1] - 2]
    input_data_outputs = input_data[1:, input_data.shape[1] - 1]

    return (input_data_inputs, input_data_outputs)

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)

    zero_proportion = counts[0] / labels.size
    one_proportion = counts[1] / labels.size

    return -1 * np.sum(
        [
            zero_proportion * np.log2(zero_proportion),
            one_proportion * np.log2(one_proportion)
        ])

def get_error_ratio(labels):
    values, counts = np.unique(labels, return_counts=True)
    print("Inspecting labels:", "values", values, "counts", counts)

    majority_vote = 0 if counts[0] > counts[1] else 1

    return counts[0] / labels.size if majority_vote == 1 else counts[1] / labels.size

def print_to_file(cmd_argument_number, lines_dictionary):
    out_file_name = sys.argv[cmd_argument_number]
    print(f"Writing to out file: {out_file_name}")
    with open(out_file_name, "w") as txt_file:
        for key in lines_dictionary:
            txt_file.write(f'{key}: {lines_dictionary[key]}\n')

_, data_labels = load_file_contents(1)
labels_entropy = entropy(data_labels)
print("Labels entropy:", labels_entropy)

error_ratio = get_error_ratio(data_labels)
print("Error ratio:", error_ratio)

print_to_file(2, { "entropy": labels_entropy, "error": error_ratio })