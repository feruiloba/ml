import sys
import numpy as np

train_input = sys.argv[1]
test_input = sys.argv[2]
train_out = sys.argv[3]
test_out = sys.argv[4]
metrics_out = sys.argv[5]

print(f"The train_input file is: {train_input}")
print(f"The test_input file is: {test_input}")
print(f"The train_out file is: {train_out}")
print(f"The test_out file is: {test_out}")
print(f"The metrics_out file is: {metrics_out}")

with open(train_input) as train_input_file:
    train_input_data = train_input_file.read()

    train_input_data_matrix = np.array(train_input_data)
    print(train_input_data_matrix.shape)
    

#print(s)
