import argparse
import matplotlib.pyplot as plt
import numpy as np
from decision_tree import load_file_contents, build_tree, predict, get_error_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help="path to training input .tsv file")
    parser.add_argument("test_input", type=str, help="path to the test input .tsv file")
    parser.add_argument("max_depth", type=int, help="maximum depth to which the tree should be built")
    args = parser.parse_args()

train_inputs, train_labels, train_headers, labels_header = load_file_contents(args.train_input)
test_inputs, test_labels, test_headers, test_labels_header = load_file_contents(args.test_input)

train_errors = []
test_errors = []
for i in range(args.max_depth):

    head_node = build_tree(train_inputs, train_labels, train_headers, 0, i)

    train_predicted_labels = predict(head_node, train_inputs, train_headers)
    train_errors = np.append(train_errors, get_error_ratio(train_predicted_labels, train_labels))

    test_predicted_labels = predict(head_node, test_inputs, test_headers)
    test_errors = np.append(test_errors, get_error_ratio(test_predicted_labels, test_labels))

fig, ax = plt.subplots()
ax.plot(range(args.max_depth), train_errors, label='error(train)')
ax.plot(range(args.max_depth), test_errors, label='error(test)')
ax.set_xlabel('max_depth')  # Add an x-label to the Axes.
ax.set_ylabel('error')  # Add a y-label to the Axes.
ax.set_title("Error rate")  # Add a title to the Axes.
ax.legend()  # Add a legend.
plt.show()

