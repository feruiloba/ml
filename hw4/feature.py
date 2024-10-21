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
    trimmed_words_dict = OrderedDict(dict())
    i = 0
    for label, sentence in dataset:
        words = sentence.split()
        trimmed_words = np.array([])
        for word in words:
            if word in glove_map.keys():
                trimmed_words = np.append(trimmed_words, word)
        trimmed_words_dict[f"{i}"] = trimmed_words
        i += 1

    return trimmed_words_dict

def glove_values(trimmed_dataset, glove_map):
    glove_dataset = np.zeros((len(trimmed_dataset), 300))

    i=0
    for sentence in trimmed_dataset.values():
        sentence_value = np.zeros(300)
        for word in sentence:
            if word in glove_map.keys():
                sentence_value = sentence_value + glove_map[word]
        glove_dataset[i] = sentence_value / sentence.size
        i += 1
        
    return glove_dataset

def print_globe_values(labels_dataset, globe_dataset, out_file_name):

    with open(out_file_name, "w") as txt_file:
        i=0
        for sentence in globe_dataset:
            globe_values_tab_separated = "\t".join(sentence.astype(str))
            txt_file.write(f"{labels_dataset[i][0]}\t{globe_values_tab_separated}\n")
            i+=1

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
    train_globe_dataset = glove_values(train_trimmed_dataset, glove_map)
    print_globe_values(train_input, train_globe_dataset, args.train_out)

    test_input = load_tsv_dataset(args.test_input)
    test_trimmed_dataset = trim(test_input, glove_map)
    test_globe_dataset = glove_values(test_trimmed_dataset, glove_map)
    print_globe_values(test_input, test_globe_dataset, args.test_out)

    val_input = load_tsv_dataset(args.validation_input)
    val_trimmed_dataset = trim(val_input, glove_map)
    val_globe_dataset = glove_values(val_trimmed_dataset, glove_map)
    print_globe_values(val_input, val_globe_dataset, args.validation_out)

