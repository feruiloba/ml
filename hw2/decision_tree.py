import argparse
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


def print_tree(Node):
    pass


def load_file_contents(input_file_name: str):
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
            one_proportion * np.log2(one_proportion),
        ]
    )


# Probably not very efficient to iterate over labels/inputs columns many times,
# but at least it doesn't grow exponentially. Plus no loops!
def mutual_information(inputs_col, labels):
    entropy_labels = entropy(labels)
    _, counts = np.unique(inputs_col, return_counts=True)
    fraction_zeros = counts[0] / inputs_col.size
    fraction_ones = counts[1] / inputs_col.size

    labels_subset_zeros = labels[np.invert(inputs_col.astype(bool))]
    labels_subset_ones = labels[inputs_col.astype(bool)]

    return entropy_labels - np.sum(
        [
            fraction_zeros * entropy(labels_subset_zeros),
            fraction_ones * entropy(labels_subset_ones),
        ]
    )


def build_tree(inputs, labels):

    biggest_mutual_info = 0
    biggest_mutual_info_col_index = 0

    for col_index in range(inputs.shape[1]):
        mutual_info_col = mutual_information(inputs[:, col_index], labels)
        print(f"mutual info for {col_index}", mutual_info_col)
        if mutual_info_col > biggest_mutual_info:
            biggest_mutual_info = mutual_info_col
            biggest_mutual_info_col_index = col_index

    print(
        "biggest mutual info col",
        biggest_mutual_info,
        "biggest_mutual_info_col_index",
        biggest_mutual_info_col_index,
    )


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

    train_inputs, train_labels = load_file_contents(args.train_input)
    build_tree(train_inputs, train_labels)

    # Here is a recommended way to print the tree to a file
    # with open(print_out, "w") as file:
    #     print_tree(dTree, file)
