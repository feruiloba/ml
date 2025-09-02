import sys
import numpy as np # type: ignore

def getInputString(inputArray):
    return "".join(np.char.array(inputArray))

def train(inputs, outputs):

    # Count the different outputs for the same input
    modeCounter = {}
    for i in range(inputs.shape[0]):
        inputString = getInputString(inputs[i])
        if (not inputString in modeCounter.keys()):
            modeCounter[inputString] = { outputs[i]: 1 }
        elif (not outputs[i] in modeCounter[inputString].keys()):
            modeCounter[inputString][outputs[i]] = 1
        else:
            modeCounter[inputString][outputs[i]] += 1

    # Get the most frequent output for the same input
    modes = {}
    for inputString, outputCountDict in modeCounter.items():
        biggestCountForInput = 0
        modeForInputString = ""
        for output, count in outputCountDict.items():
            if count > biggestCountForInput:
                modeForInputString = output
                biggestCountForInput = count

        modes[inputString] = modeForInputString

    return modes

def predict(inputs, majority_votes_dict):
    outputs = np.array([])
    for input in inputs:
        inputString = getInputString(input)
        if (inputString in majority_votes_dict.keys()):
            outputs = np.append(outputs, majority_votes_dict[inputString])
        else:
            outputs = np.append(outputs, 1)

    return outputs

# Training

train_input = sys.argv[1]
print(f"The train_input file is: {train_input}")
train_input_data = np.genfromtxt(fname=train_input, delimiter="\t", dtype=None, encoding=None)
train_input_data_inputs = train_input_data[1:,0:train_input_data.shape[1] - 2]
train_input_data_outputs = train_input_data[1:,train_input_data.shape[1] - 1]
modeMap = train(train_input_data_inputs, train_input_data_outputs)

train_predictions = predict(train_input_data_inputs, modeMap)

train_out = sys.argv[3]
print(f"Writing to train_out file: {train_out}")
with open(train_out, "w") as txt_file:
    for line in train_predictions:
        txt_file.write(" ".join(line) + "\n")

# Testing

test_input = sys.argv[2]
print(f"The test_input file is: {test_input}")
test_input_data = np.genfromtxt(fname=test_input, delimiter="\t", dtype=None, encoding=None)
test_input_data_inputs = test_input_data[1:,0:test_input_data.shape[1] - 2]

test_predictions = predict(test_input_data_inputs, modeMap)

test_out = sys.argv[4]
print(f"Writing to test_out file: {test_out}")
with open(test_out, "w") as txt_file:
    for line in test_predictions:
        txt_file.write(" ".join(line) + "\n")

# Metrics

def get_error_ratio(predicted_outputs, real_outputs):
    errorCount = 0

    for i in range(predicted_outputs.size):
        if predicted_outputs[i] != real_outputs[i]:
            errorCount += 1

    return f'{(errorCount / predicted_outputs.size):.6f}'

train_error = get_error_ratio(train_predictions, train_input_data_outputs)

test_input_data_outputs = test_input_data[1:,test_input_data.shape[1] - 1]
test_error = get_error_ratio(test_predictions, test_input_data_outputs)

metrics_out = sys.argv[5]
print(f"Writing to metrics_out file: {metrics_out}")
with open(metrics_out, "w") as txt_file:
    txt_file.write(f'{train_error}\n')
    txt_file.write(test_error)
