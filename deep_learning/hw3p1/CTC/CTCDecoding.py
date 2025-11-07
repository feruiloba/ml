import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        max_symbol = None
        previous_index = None
        for t in range(y_probs.shape[1]):
            max_index = np.argmax(y_probs[:,t])
            max_prob = y_probs[max_index][t]
            path_prob *= max_prob
            if max_index != blank and max_index != previous_index:
                max_symbol = self.symbol_set[max_index -1]
                previous_index = max_index
                decoded_path.append(max_symbol)

        decoded_path = ''.join(decoded_path)

        return decoded_path, path_prob

class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        num_timesteps = y_probs.shape[1]
        num_symbols = y_probs.shape[0]
        bestPath, FinalPathScore = None, None

        # TODO:
        # Implement the beam search decoding algorithm. This typically involves:
        # 1. Initializing a set of paths with their probabilities.
        # 2. For each time step, extending existing paths with all possible symbols.
        # 3. Merging paths that end in the same symbol.
        # 4. Pruning the set of paths to keep only the top 'beam_width' paths.
        # 5. After iterating through all time steps, selecting the best path
        #    and its score.

        def get_symbol_path(index, path=None):
            # if the symbol is blank or the previous symbol is the same, then it's the same path
            current_symbol = self.symbol_set[index-1] if index > 0 else '-'
            previous_symbol = path[-1] if path != '' else ''

            # collapse subsequent symbols
            if (current_symbol == previous_symbol):
                return ''

            if (previous_symbol == '-'):
                return previous_symbol

            return current_symbol

        def get_top_paths(paths_dict):
            return dict(sorted(paths_dict.items(), key=lambda path_prob: path_prob[1], reverse=True)[:self.beam_width])

        activePaths = {
            '': 1
        }

        tempPaths = {}

        for t in range(num_timesteps):
            activePaths = get_top_paths(activePaths)
            for path, prob in activePaths.items():
                for s in range(num_symbols):
                    symbol = get_symbol_path(s, path)
                    new_path = path + symbol
                    new_prob = prob * y_probs[s][t]
                    if new_path in tempPaths:
                        tempPaths[new_path] += new_prob
                    else:
                        tempPaths[new_path] = new_prob
            activePaths = tempPaths
            tempPaths = {}

        bestPath = ''
        bestProb = 0
        mergedPaths = {}
        for path, prob in activePaths.items():
            path = path.strip()
            if path in mergedPaths:
                mergedPaths[path] += prob
            else:
                mergedPaths[path] = prob
            if prob > bestProb:
                bestPath = path
                bestProb = prob

        return bestPath, mergedPaths


