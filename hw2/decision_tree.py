import argparse
from collections import OrderedDict
import sys
import numpy as np  # type: ignore


class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree.
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired
    """

    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.header = None
        self.mutual_info = None

def print_tree(Node):
    pass


def load_file_contents(input_file_name: str):
    print(f"Loading input file: {input_file_name}")
    input_data = np.genfromtxt(
        fname=input_file_name, delimiter="\t", dtype=None, encoding=None
    )
    input_data_inputs = input_data[1:, 0 : input_data.shape[1] - 2]
    input_data_outputs = input_data[1:, input_data.shape[1] - 1]
    input_data_headers = input_data[0,:]

    return (input_data_inputs, input_data_outputs, input_data_headers)


def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)

    zero_proportion = counts[0] / labels.size if counts.any() else 0
    one_proportion = counts[1] / labels.size if counts.any() and counts.size > 1 else 0

    return -1 * np.sum(
        [
            zero_proportion * np.log2(zero_proportion) if zero_proportion > 0 else 0,
            one_proportion * np.log2(one_proportion) if one_proportion > 0 else 0,
        ]
    )

def split(col_to_split, inputs):
    col_to_split_bools = col_to_split.astype(bool)
    inputs_when_zero = inputs[np.invert(col_to_split_bools)]
    inputs_when_one = inputs[col_to_split_bools]

    return (inputs_when_zero, inputs_when_one)

# Probably not very efficient to iterate over labels/inputs columns many times,
# but at least it doesn't grow exponentially. Plus no loops!
def mutual_information(inputs_col, labels):
    entropy_labels = entropy(labels)
    _, counts = np.unique(inputs_col, return_counts=True)

    fraction_zeros = counts[0] / inputs_col.size if counts.any() else 0
    fraction_ones = counts[1] / inputs_col.size if counts.any() and counts.size > 1 else  0

    labels_for_zero_input, labels_for_one_input = split(inputs_col, labels)

    return entropy_labels - np.sum(
        [
            fraction_zeros * entropy(labels_for_zero_input),
            fraction_ones * entropy(labels_for_one_input),
        ]
    )

def majority_vote(labels):
    values, counts = np.unique(labels, return_counts=True)
    if values.size == 0:
        return None
    elif values.size == 1:
        return values[0]
    else:
        return values[0] if counts[0] > counts[1] else values[1]

def build_tree(inputs, labels, headers, depth, max_depth):

    node = Node()
    node.attr = inputs
    node.vote = majority_vote(labels)

    if (depth >= max_depth or inputs.shape[1] <= 1):
        return node

    mutual_informations = {}
    for col_index in range(inputs.shape[1]):
         mutual_informations[col_index] = mutual_information(inputs[:, col_index], labels)

    # https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/
    sorted_mutual_infos = OrderedDict({key:value for key, value in sorted(mutual_informations.items(), key=lambda mutual_informations: mutual_informations[1])})

    # print("mutual_infos", sorted_mutual_infos)

    col_to_split = sorted_mutual_infos.popitem()
    node.mutual_info = col_to_split[1]
    node.header = headers[col_to_split[0]]
    new_headers = np.delete(headers, col_to_split[0], 0)

    # print(f"splitting by column {headers[col_to_split[0]]} with MI {col_to_split[1]}")

    inputs_with_zero, inputs_with_one = split(inputs[:,col_to_split[0]], inputs)
    labels_with_zero, labels_with_one = split(inputs[:,col_to_split[0]], labels)

    # print("inputs shape", inputs.shape)
    # print("inputs with zero shape", inputs_with_zero.shape)
    # print("labels with zero shape", labels_with_zero.shape)
    # print("inputs with one shape", inputs_with_one.shape)
    # print("labels with one shape", labels_with_one.shape)

    node.right = build_tree(inputs_with_zero, labels_with_zero, new_headers, depth + 1, max_depth)
    node.left = build_tree(inputs_with_one, labels_with_one, new_headers, depth + 1, max_depth)

    return node

if __name__ == "__main__":
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_input", type=str, help="path to training input .tsv file"
    )
    parser.add_argument("test_input", type=str, help="path to the test input .tsv file")
    parser.add_argument(
        "max_depth", type=int, help="maximum depth to which the tree should be built"
    )
    parser.add_argument(
        "train_out",
        type=str,
        help="path to output .txt file to which the feature extractions on the training data should be written",
    )
    parser.add_argument(
        "test_out",
        type=str,
        help="path to output .txt file to which the feature extractions on the test data should be written",
    )
    parser.add_argument(
        "metrics_out",
        type=str,
        help="path of the output .txt file to which metrics such as train and test error should be written",
    )
    parser.add_argument(
        "print_out",
        type=str,
        help="path of the output .txt file to which the printed tree should be written",
    )
    args = parser.parse_args()

    # Here's an example of how to use argparse
    print_out = args.print_out

    train_inputs, train_labels, headers = load_file_contents(args.train_input)
    head_node = build_tree(train_inputs, train_labels, headers, 0, 3)

    # Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)
