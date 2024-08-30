import sys
import numpy as np # type: ignore

# Training inputs
train_input = sys.argv[1]
print(f"The train_input file is: {train_input}")
train_input_data = np.genfromtxt(fname=train_input, delimiter="\t", dtype=None, encoding=None)
train_input_data_headers = train_input_data[0]
train_input_data_inputs = train_input_data[1:,0:train_input_data_headers.size-2]
train_input_data_outputs = train_input_data[1:,train_input_data_headers.size-1]

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
            if biggestCountForInput < count:
                modeForInputString = output
        modes[inputString] = modeForInputString

    return modes

# Testing inputs
test_input = sys.argv[2]
print(f"The test_input file is: {test_input}")
test_input_data = np.genfromtxt(fname=train_input, delimiter="\t", dtype=None, encoding=None)
test_input_data_headers = test_input_data[0]
test_input_data_inputs = test_input_data[1:,0:train_input_data_headers.size-2]
test_input_data_outputs = test_input_data[1:,train_input_data_headers.size-1]


train_out = sys.argv[3]
test_out = sys.argv[4]
metrics_out = sys.argv[5]

print(f"The train_out file is: {train_out}")
print(f"The test_out file is: {test_out}")
print(f"The metrics_out file is: {metrics_out}")

modeMap = train(train_input_data_inputs, train_input_data_outputs)

def predict(inputs, majority_votes_dict):
    outputs = np.array([])
    for i in range(inputs.shape[0]):
        inputString = getInputString(inputs[i])
        print(inputString)
        if (inputString in majority_votes_dict.keys()):
            outputs = np.append(outputs, majority_votes_dict[inputString])
        else:
            outputs = np.append(outputs, 1)

    return outputs

print(predict(test_input_data_inputs, modeMap))