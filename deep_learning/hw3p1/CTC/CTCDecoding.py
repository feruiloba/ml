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
        self.blank_index = 0

    def initialize_paths(self, y_t):
            paths_blank = {""}
            paths_symbol = set()
            blank_path_score = {"": y_t[self.blank_index]}
            path_score = {}

            for i in range(len(self.symbol_set)):
                symbol = self.symbol_set[i]
                paths_symbol.add(symbol)
                path_score[symbol] = y_t[i + 1]  # offset +1 for non-blank

            return paths_blank, paths_symbol, blank_path_score, path_score

    def prune(self, paths_blank, paths_symbol, blank_probs, path_probs):
        all_scores = list(blank_probs.values()) + list(path_probs.values())
        top_probabilities = sorted(all_scores, reverse=True)[:self.beam_width]
        min_prob = top_probabilities[-1]

        pruned_blank = set()
        pruned_symbol = set()
        pruned_blank_scores = {}
        pruned_path_scores = {}

        for path in paths_blank:
            if blank_probs[path] >= min_prob:
                pruned_blank.add(path)
                pruned_blank_scores[path] = blank_probs[path]

        for path in paths_symbol:
            if path_probs[path] >= min_prob:
                pruned_symbol.add(path)
                pruned_path_scores[path] = path_probs[path]

        return pruned_blank, pruned_symbol, pruned_blank_scores, pruned_path_scores

    def extend_with_blank(self, paths_blank, paths_symbol, blank_probs, path_probs, y_t):
        updated_blank = set()
        updated_blank_scores = {}

        for path in paths_blank:
            updated_blank.add(path)
            updated_blank_scores[path] = blank_probs[path] * y_t[self.blank_index]

        for path in paths_symbol:

            if path in updated_blank:
                updated_blank_scores[path] += path_probs[path] * y_t[self.blank_index]
            else:
                updated_blank.add(path)
                updated_blank_scores[path] = path_probs[path] * y_t[self.blank_index]

        return updated_blank, updated_blank_scores

    def merge_identical(self, paths_blank, blank_probs, paths_symbol, path_probs):
        merged = set(paths_symbol)
        final_probs = dict(path_probs)
        for p in paths_blank:
            if p in merged:
                final_probs[p] += blank_probs[p]
            else:
                merged.add(p)
                final_probs[p] = blank_probs[p]

        return merged, final_probs

    def extend_with_symbol(self, paths_blank, paths_symbol, blank_probs, path_probs, y_t):
        updated_symbol = set()
        updated_probs = {}

        # Extend paths ending with blank
        for p in paths_blank:
            for i in range(len(self.symbol_set)):
                new_path = p + self.symbol_set[i]
                updated_symbol.add(new_path)
                updated_probs[new_path] = blank_probs[p] * y_t[i + 1]

        # Extend paths ending with symbols
        for p in paths_symbol:
            for i in range(len(self.symbol_set)):
                symbol = self.symbol_set[i]
                new_path = p if symbol == p[-1] else p + symbol

                if new_path in updated_symbol:
                    updated_probs[new_path] += path_probs[p] * y_t[i + 1]
                else:
                    updated_symbol.add(new_path)
                    updated_probs[new_path] = path_probs[p] * y_t[i + 1]

        return updated_symbol, updated_probs

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
        T = y_probs.shape[1]

        # TODO:
        # Implement the beam search decoding algorithm. This typically involves:
        # 1. Initializing a set of paths with their probabilities.
        # 2. For each time step, extending existing paths with all possible symbols.
        # 3. Merging paths that end in the same symbol.
        # 4. Pruning the set of paths to keep only the top 'beam_width' paths.
        # 5. After iterating through all time steps, selecting the best path
        #    and its score.

        paths_blank, paths_symbol, blank_probs, path_probs = self.initialize_paths(y_probs[:, 0])

        for t in range(1, T):
            paths_blank, paths_symbol, blank_probs, path_probs = self.prune(
                paths_blank, paths_symbol, blank_probs, path_probs
            )

            new_blank, new_blank_probs = self.extend_with_blank(
                paths_blank, paths_symbol, blank_probs, path_probs, y_probs[:, t]
            )

            new_symbol, new_symbol_scores = self.extend_with_symbol(
                paths_blank, paths_symbol, blank_probs, path_probs, y_probs[:, t]
            )

            paths_blank = new_blank
            paths_symbol = new_symbol
            blank_probs = new_blank_probs
            path_probs = new_symbol_scores

        merged_paths, final_scores = self.merge_identical(paths_blank, blank_probs, paths_symbol, path_probs)

        best_path = max(final_scores, key=final_scores.get)
        return best_path, final_scores