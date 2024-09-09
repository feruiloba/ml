"""
Create and write a program inspection.py to calculate the label entropy at the root
(i.e. the entropy of the labels before any splits) and the error rate (the percent of incorrectly
classified instances) of classifying using a majority vote (picking the label with the most
examples). You do not need to look at the values of any of the attributes to do these calculations;
knowing the labels of each example is sufficient. Entropy should be calculated in bits using log
base 2
"""

import sys

import numpy as np


def load_file_contents(cmd_argument_number: int):
    input_file_name = sys.argv[cmd_argument_number]
    print(f"Loading input file: {input_file_name}")
    input_data = np.genfromtxt(fname=input_file_name, delimiter="\t", dtype=None, encoding=None)
    input_data_inputs = input_data[1:,0:input_data.shape[1] - 2]
    input_data_outputs = input_data[1:,input_data.shape[1] - 1]

    return (input_data_inputs, input_data_outputs)


