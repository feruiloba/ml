import argparse
from collections import OrderedDict
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
        self.stats_zeros = None
        self.stats_ones = None

def load_file_contents(input_file_name: str):
    print(f"Loading input file: {input_file_name}")
    input_data = np.genfromtxt(
        fname=input_file_name, delimiter="\t", dtype=None, encoding=None
    )
    input_data_inputs = input_data[1:, 0 : input_data.shape[1] - 1]
    input_data_outputs = input_data[1:, input_data.shape[1] - 1]
    input_data_headers = input_data[0, 0 : input_data.shape[1] - 1]
    labels_header = input_data[0, input_data.shape[1] - 1]

    return (input_data_inputs, input_data_outputs, input_data_headers, labels_header)

def get_proportions(inputs):
    values, counts = np.unique(inputs, return_counts=True)

    if not counts.any():
        zero_proportion = 0
        one_proportion = 0
    elif counts.size == 1:
        if values[0] == "0":
            zero_proportion = counts[0] / inputs.size
            one_proportion = 0
        else:
            zero_proportion = 0
            one_proportion = counts[0] / inputs.size
    else:
        zero_proportion = counts[0] / inputs.size
        one_proportion = counts[1] / inputs.size

    return (zero_proportion, one_proportion)

def entropy(labels):
    zero_proportion, one_proportion = get_proportions(labels)

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
    zero_proportion, one_proportion = get_proportions(inputs_col)
    labels_for_zero_input, labels_for_one_input = split(inputs_col, labels)

    return entropy_labels - np.sum(
        [
            zero_proportion * entropy(labels_for_zero_input),
            one_proportion * entropy(labels_for_one_input),
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

def stats(labels):
    values, counts = np.unique(labels, return_counts=True)
    if not counts.any():
        return f"[0 0/0 1]"
    elif counts.size == 1:
        if values[0] == "0":
            return f"[{counts[0]} 0/0 1]"
        else:
            return f"[0 0/{counts[0]} 1]"
    else:
        return f"[{counts[0]} 0/{counts[1]} 1]"

def are_all_inputs_the_same(inputs):
    _, counts = np.unique(inputs, return_counts=True)
    return (not counts.any()) or counts[0] == 0 or inputs.size == counts[0]

def build_tree(inputs, labels, headers, depth, max_depth):

    node = Node()
    node.attr = inputs
    node.vote = majority_vote(labels)

    if (depth >= max_depth or inputs.shape[1] < 1 or are_all_inputs_the_same(labels) or are_all_inputs_the_same(inputs)):
        return node

    mutual_informations = {}
    for col_index in range(inputs.shape[1]):
        mutual_informations[col_index] = mutual_information(inputs[:, col_index], labels)

    # https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/
    # tie breaker: https://stackoverflow.com/questions/54300715/python-3-list-sorting-with-a-tie-breaker
    sorted_mutual_infos = OrderedDict(
        {
            key:value for key, value in sorted(
                mutual_informations.items(),
                key=lambda mutual_informations: (mutual_informations[1], -mutual_informations[0]))
        })

    col_to_split_index, col_mutual_info = sorted_mutual_infos.popitem()

    node.mutual_info = col_mutual_info
    node.header = headers[col_to_split_index]
    new_headers = np.delete(headers, col_to_split_index, 0)

    col_to_split = inputs[:,col_to_split_index]
    new_inputs = np.delete(inputs, col_to_split_index, 1)
    inputs_with_zero, inputs_with_one = split(col_to_split, new_inputs)
    labels_with_zero, labels_with_one = split(col_to_split, labels)
    node.stats_zeros = stats(labels_with_zero)
    node.stats_ones = stats(labels_with_one)

    node.left = build_tree(inputs_with_zero, labels_with_zero, new_headers, depth + 1, max_depth)
    node.right = build_tree(inputs_with_one, labels_with_one, new_headers, depth + 1, max_depth)

    return node

def predict_row_label(node: Node, inputs, headers):

        if (node.left == None and node.right == None):
            return node.vote

        # find the item that the node is talking about in the inputs
        input_index = np.where(headers == node.header)[0]
        if (inputs[input_index] == "0"):
            return predict_row_label(node.left, inputs, headers)
        else:
            return predict_row_label(node.right, inputs, headers)

def predict(node, inputs, headers):
    predicted_labels = []
    for row in inputs:
        predicted_labels = np.append(predicted_labels, predict_row_label(node, row, headers))

    return predicted_labels

def get_error_ratio(predicted_outputs, real_outputs):
    errorCount = 0

    for i in range(predicted_outputs.size):
        if predicted_outputs[i] != None and int(predicted_outputs[i]) != int(real_outputs[i]):
            errorCount += 1

    return errorCount / predicted_outputs.size

def print_tree_inner(node, depth = 0):
    if node.left == None and node.right == None:
        return ""

    depth_string = "| " * (depth + 1)

    return (f"\n{depth_string} {node.header} = 0: {node.stats_zeros}" + print_tree_inner(node.left, depth + 1)
            + f"\n{depth_string} {node.header} = 1: {node.stats_ones}" + print_tree_inner(node.right, depth + 1))

def print_tree(node, labels):
    return stats(labels) + print_tree_inner(node)

def print_to_file(print_out, content):
        print(f"Writing to out file {print_out}")
        with open(print_out, "w") as txt_file:
            for line in content:
                txt_file.write(str(line) + "\n")

def print_tree_to_file(print_out, content):
    print(f"Writing to out file {print_out}")
    with open(print_out, "w") as txt_file:
        txt_file.write(str(content) + "\n")

def print_metrics_to_file(print_out, train_error, test_error):
    print(f"Writing to out file {print_out}")
    with open(print_out, "w") as txt_file:
        txt_file.write(f'error(train): {train_error}\n')
        txt_file.write(f'error(test): {test_error}\n')

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

    train_inputs, train_labels, train_headers, labels_header = load_file_contents(args.train_input)
    # print("train_inputs.shape", train_inputs.shape, "train_labels.shape", train_labels.shape, "train_headers.shape", train_headers.shape, "labels_header", labels_header)

    head_node = build_tree(train_inputs, train_labels, train_headers, 0, args.max_depth)

    train_predicted_labels = predict(head_node, train_inputs, train_headers)
    print_to_file(args.train_out, train_predicted_labels)

    test_inputs, test_labels, test_headers, test_labels_header = load_file_contents(args.test_input)
    test_predicted_labels = predict(head_node, test_inputs, test_headers)
    print_to_file(args.test_out, test_predicted_labels)

    train_error_ratio = get_error_ratio(train_predicted_labels, train_labels)
    test_error_ratio = get_error_ratio(test_predicted_labels, test_labels)
    print_metrics_to_file(args.metrics_out, train_error_ratio, test_error_ratio)

    printed_tree = print_tree(head_node, train_labels)
    print_tree_to_file(args.print_out, printed_tree)
    print(printed_tree)
