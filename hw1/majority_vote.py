import sys
import numpy as np # type: ignore

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

# Obtained from https://stackoverflow.com/questions/49522800/numpy-read-tsv-file-as-ndarray
train_input_data = np.genfromtxt(fname=train_input, delimiter="\t", skip_header=0, filling_values=1)

num_rows, num_cols = train_input_data.shape
for i in range(num_rows):
    for j in range(num_cols):
        print("i", i, "j", j, "data", train_input_data[i][j])


#print(s)
