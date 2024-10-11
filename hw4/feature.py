import csv
from typing import OrderedDict
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


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
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8', dtype='l,O')
    return dataset

def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def trim(dataset, glove_map):
    trimmed_dataset = OrderedDict(dict[int, np.array]())
    for sentence_index, (sentiment, sentence) in enumerate(dataset):
        words = sentence.split()
        for word in words:
            if word in glove_map.keys():
                word_value = dict({ "name": word, "value": glove_map[word]})
                if sentence_index in trimmed_dataset:
                    new_sentence = np.append(trimmed_dataset[sentence_index], word_value)
                else:
                    new_sentence = np.array(word_value, dtype=dict)
                
                trimmed_dataset[sentence_index] = new_sentence

    return trimmed_dataset

def glove_values(trimmed_dataset):
    # print(" blas")
    # gloved_sentence = dict()
    # sentence, word, name
    # print(trimmed_dataset[1][0]["name"])
    glove_dataset = OrderedDict(dict[int, np.array]())

    for sentence_index, sentence in trimmed_dataset.items():
        sentence_values = np.zeros(shape=sentence[0]["value"].shape)
        for word in sentence:
            sentence_values = sentence_values + word["value"]
        
        sentence_values = sentence_values / sentence.size
        glove_dataset[sentence_index] = sentence_values

    return glove_dataset
        
def print_globe_values(labels_dataset, globe_dataset, out_file_name):

    with open(out_file_name, "w") as txt_file: 
        for index, sentence in globe_dataset.items():
            globe_values_tab_separated = " ".join(sentence.astype(str))
            txt_file.write(f"{labels_dataset[index][0]}\t{globe_values_tab_separated}\n")

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str,  help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    glove_map = load_feature_dictionary(args.feature_dictionary_in)

    train_input = load_tsv_dataset(args.train_input)
    train_trimmed_dataset = trim(train_input, glove_map)
    train_globe_dataset = glove_values(train_trimmed_dataset)
    print_globe_values(train_input, train_globe_dataset, args.train_out)
    
    test_input = load_tsv_dataset(args.test_input)
    test_trimmed_dataset = trim(test_input, glove_map)
    test_globe_dataset = glove_values(test_trimmed_dataset)
    print_globe_values(test_input, test_globe_dataset, args.test_out)

    val_input = load_tsv_dataset(args.validation_input)
    val_trimmed_dataset = trim(val_input, glove_map)
    val_globe_dataset = glove_values(val_trimmed_dataset)
    print_globe_values(test_input, test_globe_dataset, args.validation_out)

    
