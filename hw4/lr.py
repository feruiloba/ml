import numpy as np
import argparse

# would normally import from other file, but this will be used in gradescope
def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8')
    return dataset

def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int,
    learning_rate : float
) -> None:

    for _ in range(0, num_epoch):

        sig_x = sigmoid(X @ theta)

        sigmoid_minus_y = (sig_x - y)

        gradient = X.T @ sigmoid_minus_y

        theta = theta - learning_rate * gradient

    return theta


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:

    predictions = np.array([])
    sigmoid_theta_x = sigmoid(X @ theta)
    prediction = (sigmoid_theta_x >= 0.5).astype(int)
    predictions = np.append(predictions, prediction)

    return predictions

def compute_error(
    y_pred : np.ndarray,
    y : np.ndarray
) -> float:
    errorCount = 0

    y_pred[y.astype(bool)]
    for i in range(y_pred.size):
        if int(y_pred[i]) != int(y[i]):
            errorCount += 1

    return errorCount / y_pred.size

def get_labels_features(dataset):
    labels = dataset[:, 0]
    features = dataset[:, 1:dataset.shape[1]]
    features_with_intercept = np.insert(features, 0, 1, 1)

    print("features_with_intercept", features_with_intercept)
    return (labels, features_with_intercept)

def print_to_file(predictions, out_file_name):
     with open(out_file_name, "w") as txt_file:
         for prediction in predictions:
             txt_file.write(f"{prediction}\n")

def print_metrics(train_error, test_error, metrics_out):
    print(f"Writing to metrics_out file: {metrics_out}")
    with open(metrics_out, "w") as txt_file:
        txt_file.write(f'error(train): {train_error}\n')
        txt_file.write(f'error(test): {test_error}\n')


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int,
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()


    train_dataset = load_tsv_dataset(args.train_input)

    num_epoch = args.num_epoch
    learning_rate = args.learning_rate

    train_y, train_X = get_labels_features(train_dataset)
    train_thetas = np.zeros(train_X.shape[1])
    train_thetas = train(theta=train_thetas, X=train_X, y=train_y, num_epoch=num_epoch, learning_rate=learning_rate)

    train_predictions = predict(train_thetas, train_X)
    print_to_file(train_predictions, args.train_out)
    train_error = compute_error(train_predictions, train_y)

    test_dataset = load_tsv_dataset(args.test_input)
    test_y, test_X = get_labels_features(test_dataset)
    test_predictions = predict(train_thetas, test_X)
    print_to_file(test_predictions, args.test_out)
    test_error = compute_error(test_predictions, test_y)

    print_metrics(train_error, test_error, args.metrics_out)