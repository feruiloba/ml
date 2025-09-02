import argparse
import matplotlib.pyplot as plt
import numpy as np
from lr import sigmoid, get_labels_features, load_tsv_dataset, train, predict, compute_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help="path to training input .tsv file")
    parser.add_argument("val_input", type=str, help="path to the test input .tsv file")
    parser.add_argument("num_epoch", type=int, help="maximum depth to which the tree should be built")
    args = parser.parse_args()

train_input = load_tsv_dataset(args.train_input)
train_y, train_X = get_labels_features(train_input)

val_input = load_tsv_dataset(args.val_input)
val_dataset = load_tsv_dataset(args.val_input)
val_y, val_X = get_labels_features(val_dataset)
learning_rate = 0.1

train_lk = np.array([], dtype=float)
val_lk = np.array([], dtype=float)
step_size = 5
for num_epoch in range(0, args.num_epoch, step_size):

    train_thetas = np.zeros(train_X.shape[1])
    train_thetas = train(theta=train_thetas, X=train_X, y=train_y, num_epoch=num_epoch, learning_rate=learning_rate)
    train_sigmoid_theta_x = - np.sum(sigmoid(train_X @ train_thetas)) / train_X.shape[1]
    train_lk = np.append(train_lk, train_sigmoid_theta_x)

    val_thetas = np.zeros(val_X.shape[1])
    val_thetas = train(theta=val_thetas, X=val_X, y=val_y, num_epoch=num_epoch, learning_rate=learning_rate)
    val_sigmoid_theta_x = - np.sum(sigmoid(val_X @ val_thetas)) / val_X.shape[1]
    val_lk = np.append(val_lk, val_sigmoid_theta_x)

fig, ax = plt.subplots()
ax.plot(range(0, args.num_epoch, step_size), train_lk, label='likelihood(train)')
ax.plot(range(0, args.num_epoch, step_size), val_lk, label='likelihood(val)')
ax.set_xlabel('num_epoch')  # Add an x-label to the Axes.
ax.set_ylabel('likelihood')  # Add a y-label to the Axes.
ax.set_title("Negative log likelihood")  # Add a title to the Axes.
ax.legend()  # Add a legend.
plt.show()

