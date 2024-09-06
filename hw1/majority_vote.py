import sys
import numpy as np # type: ignore

def train(outputs):
    zero_counter = 0
    one_counter = 0
    for i in range(outputs.size):
        if (outputs[i] == 0):
            zero_counter += 1
        else:
            one_counter += 1

    return 0 if zero_counter > one_counter else 1

def predict(inputs, majority_vote):
    outputs = np.array([])
    for _ in inputs:
        outputs = np.append(outputs, int(majority_vote))

    return outputs

# Training
train_input = sys.argv[1]
print(f"The train_input file is: {train_input}")
train_input_data = np.genfromtxt(fname=train_input, delimiter="\t", dtype=None, encoding=None)
train_input_data_inputs = train_input_data[1:,0:train_input_data.shape[1] - 2]
train_input_data_outputs = train_input_data[1:,train_input_data.shape[1] - 1]

majority_vote_output = train(train_input_data_outputs)

train_predictions = predict(train_input_data_inputs, majority_vote_output)

train_out = sys.argv[3]
print(f"Writing to train_out file: {train_out}")
with open(train_out, "w") as txt_file:
    for line in train_predictions:
        txt_file.write(str(line) + "\n")

# Testing

test_input = sys.argv[2]
print(f"The test_input file is: {test_input}")
test_input_data = np.genfromtxt(fname=test_input, delimiter="\t", dtype=None, encoding=None)
test_input_data_inputs = test_input_data[1:,0:test_input_data.shape[1] - 2]

test_predictions = predict(test_input_data_inputs, majority_vote_output)

test_out = sys.argv[4]
print(f"Writing to test_out file: {test_out}")
with open(test_out, "w") as txt_file:
    for line in test_predictions:
        txt_file.write(str(line) + "\n")

# Metrics

def get_error_ratio(predicted_outputs, real_outputs):
    errorCount = 0

    for i in range(predicted_outputs.size):
        if int(predicted_outputs[i]) != int(real_outputs[i]):
            errorCount += 1

    return f'{(errorCount / predicted_outputs.size):.6f}'

train_error = get_error_ratio(train_predictions, train_input_data_outputs)

test_input_data_outputs = test_input_data[1:,test_input_data.shape[1] - 1]
test_error = get_error_ratio(test_predictions, test_input_data_outputs)

metrics_out = sys.argv[5]
print(f"Writing to metrics_out file: {metrics_out}")
with open(metrics_out, "w") as txt_file:
    txt_file.write(f'error(train): {train_error}\n')
    txt_file.write(f'error(test): {test_error}\n')

