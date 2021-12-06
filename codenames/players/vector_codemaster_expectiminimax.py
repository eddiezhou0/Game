from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import scipy.spatial.distance
import itertools
import operator
import heapq
from typing import Tuple, List

from players.codemaster import *

class ExpectiMiniMaxCodemaster(Codemaster):
    def __init__(self, **kwargs):
        """Set up word list and handle pretrained vectors"""
        super().__init__()

        glove_vecs = kwargs.get("glove_vecs", None)
        word_vectors = kwargs.get("word_vectors", None)

        self.all_vectors = []
        if glove_vecs is not None:
            self.all_vectors.append(glove_vecs)
        if word_vectors is not None:
            self.all_vectors.append(word_vectors)
        if "vectors" in kwargs:
            for vecs in kwargs["vectors"]:
                self.all_vectors.append(vecs)

        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer()
        self.word_distances = None
        self.words_on_board = None
        self.key_grid = None
        self.good_color = kwargs['color']
        self.bad_color = self._other_color(self.good_color)
        self.max_clue_num = 3
        self.max_depth = 1

        # Potential codemaster clues
        self.cm_word_set = set([])
        with open('players/cm_wordlist.txt') as infile:
            for line in infile:
                self.cm_word_set.add(line.rstrip().lower())

    def set_game_state(self, words_on_board: List[str], key_grid: List[str]) -> None:
        """A set function for wordOnBoard and keyGrid (called 'map' in framework) """
        self.words_on_board = words_on_board.copy()
        self.key_grid = key_grid

        # intialize word distances on first call
        if self.word_distances is None:
            self._calc_distance_between_words_on_board_and_clue()

    def _calc_distance_between_words_on_board_and_clue(self) -> None:
        """Create word-distance dictionaries for both red words and bad words"""
        self.word_distances = {}
        for word in self.words_on_board:
            print("Calculating distance for " + word)
            self.word_distances[word] = {}
            word_stacked = self._hstack_word_vectors(word.lower())
            for potentialClue in self.cm_word_set:
                try:
                    dist = scipy.spatial.distance.cosine(word_stacked, self._hstack_word_vectors(potentialClue))
                except TypeError:
                    dist = np.inf
                self.word_distances[word][potentialClue] = dist

    def _calc_distance_between_words_and_potential_clues(self, word_set):
        word_distances = {}
        for potentialClue in word_set:
            word_distances[potentialClue] = {}
            for word in self.words_on_board:
                # print("Calculating distance for " + word)
                word_stacked = self._hstack_word_vectors(word.lower())
                try:
                    dist = scipy.spatial.distance.cosine(word_stacked, self._hstack_word_vectors(potentialClue))
                except TypeError:
                    dist = np.inf
                word_distances[potentialClue][word] = dist
        return word_distances

    def get_clue(self) -> Tuple[str, int]:
        """Function that returns a clue word and number of estimated related words on the board"""
        clueNum, val = self._expectiminimax(max, self.good_color, self.max_depth, self.words_on_board)
        # clueNum, val = self._expectiminimax_ab(self.good_color, self.max_depth, self.words_on_board, float('-inf'), float('inf'))
        clue, num = clueNum

        # Debugging values
        print(f"The {self.good_color} clue is {clue} {num}")
        print("Old board + Expected new board")
        print(self.words_on_board)
        new_board = self._simulate_guesses(self.good_color, clue, num, self.words_on_board)
        print(new_board)
        print(self._is_game_over(new_board))
        return (clue, num)

    def _expectiminimax_ab(self, color, depth, words_on_board, a, b) -> Tuple[str, int]:

        if depth == 0 or self._is_game_over(words_on_board):
            # TODO change self.good_color to actual last player
            value = self._heuristicFunction(self._other_color(color), words_on_board)
            return (None, value)

        # Progress indicator
        if depth == self.max_depth:
            num_clues = len(self.cm_word_set)
            progress_counter = 0

        best_clue = None
        if color == self.good_color:
            # maximize
            value = float('-inf')
            for potentialClue in self.cm_word_set:

                if depth == self.max_depth:
                    progress_counter += 1
                    self.printProgressBar(progress_counter, num_clues, prefix = f'{color} Progress:', suffix = 'Complete', length = 50)

                for num in range(1, self.max_clue_num + 1):
                    new_words_on_board = self._simulate_guesses(color, potentialClue, num, words_on_board)
                    _, new_value = self._expectiminimax_ab(self._other_color(color), depth-1, new_words_on_board, a, b)
                    if new_value > value:
                        best_clue = (potentialClue, num)
                        value = new_value
                    if value >= b:
                        return (best_clue, value)
                    a = max(a, value)
            return (best_clue, value)
        else:
            # minimize
            value = float('inf')
            for potentialClue in self.cm_word_set:
                for num in range(1, self.max_clue_num + 1):
                    new_words_on_board = self._simulate_guesses(color, potentialClue, num, words_on_board)
                    _, new_value = self._expectiminimax_ab(self._other_color(color), depth-1, new_words_on_board, a, b)
                    if new_value < value:
                        best_clue = (potentialClue, num)
                        value = new_value
                    if value <= a:
                        return (best_clue, value)
                    b = min(b, value)
            return (best_clue, value)

    def _expectiminimax(self, function, color, depth, words_on_board) -> Tuple[str, int]:

        if depth == 0 or self._is_game_over(words_on_board):
            # TODO change self.good_color to actual last player
            value = self._heuristicFunction(self._other_color(color), words_on_board)
            return (None, value)

        # TODO prune word set for illegal clues
        if depth == 2:
            CURSOR_UP_ONE = '\x1b[1A'
            ERASE_LINE = '\x1b[2K'
            num_clues = len(self.cm_word_set)
            progress_counter = 0
        clueOutcomes = {}
        for potentialClue in self.cm_word_set:
            if depth == 2:
                progress_counter += 1
                #print(CURSOR_UP_ONE + ERASE_LINE)
                print(f"{progress_counter}/{num_clues} : {progress_counter/num_clues * 100}%")
            for num in range(1, self.max_clue_num + 1):
                # TODO alpha-beta pruning
                # TODO handle end-game states
                new_words_on_board = self._simulate_guesses(color, potentialClue, num, words_on_board)
                next_function = self._weightedavg if function is max else max
                _, value = self._expectiminimax(next_function, self._other_color(color), depth-1, new_words_on_board)
                clueOutcomes[(potentialClue, num)] = value

        if function is max:
            return function(clueOutcomes.items(), key=operator.itemgetter(1))
        else:
            return function(clueOutcomes)

    def _weightedavg(self, clueOutcomes):
        clueProbs = {}
        potentialClueNums = []
        potentialClues = []
        for clueNum in clueOutcomes:
            potentialClueNums.append(clueNum)
            potentialClues.append(clueNum[0])

        word_distances = self._calc_distance_between_words_and_potential_clues(potentialClues)
        for potentialClueNum in potentialClueNums:
            clueDists = word_distances[potentialClueNum[0]]
            distSum = sum(clueDists.values())
            average = distSum / len(clueDists)
            clueProbs[potentialClueNum] = average

        factor = 1.0/sum(clueProbs.values())
        for clue in clueProbs:
            clueProbs[clue] *= factor

        return "", sum(clueProbs.values()) / len(clueProbs)

    def _simulate_guesses(self, guesser_color, clue, num, words_on_board):

        # use heapq for priority queue
        best_words = []
        new_words_on_board = words_on_board.copy()

        # find and sort best words
        # TODO cache the best words
        for word_index in range(len(words_on_board)):
            word = new_words_on_board[word_index]
            if word[0] == '*':
                continue
            word_distance = self.word_distances[word][clue]
            heapq.heappush(best_words, (word_distance, word_index))

        # try best words
        for i in range(num):
            _, word_index = heapq.heappop(best_words)
            word_color = self.key_grid[word_index]
            new_words_on_board[word_index] = "*" + word_color + "*"
            if self.key_grid[word_index] != guesser_color:
                break

        return new_words_on_board

    def _heuristicFunction(self, last_player_color, words_on_board):
        good_remaining = 0
        bad_remaining = 0
        neutral_remaining = 0
        assasin_remaining = 0

        for i in range(len(self.key_grid)):
            # words on board that have already been identified will have been replaced with *<operatorName>*
            # so the first character is '*'
            if words_on_board[i][0] == '*':
                continue
            elif self.key_grid[i] == self.good_color:
                good_remaining += 1
            elif self.key_grid[i] == self.bad_color:
                bad_remaining += 1
            elif self.key_grid[i] == "Civilian":
                neutral_remaining += 1
            elif self.key_grid[i] == "Assassin":
                assasin_remaining += 1
            else:
                # TODO should never get here, throw error
                pass

        if good_remaining == 0:
            return 10 + bad_remaining

        if bad_remaining == 0:
            return -10 - good_remaining

        if assasin_remaining == 0 and last_player_color == self.good_color:
            return -10 - good_remaining
        elif assasin_remaining == 0 and last_player_color == self.bad_color:
            return 10 + bad_remaining

        return bad_remaining - good_remaining

    def _is_game_over(self, words_on_board):
        good_remaining = 0
        bad_remaining = 0
        neutral_remaining = 0
        assasin_remaining = 0

        for i in range(len(self.key_grid)):
            # words on board that have already been identified will have been replaced with *<operatorName>*
            # so the first character is '*'
            if words_on_board[i][0] == '*':
                continue
            elif self.key_grid[i] == self.good_color:
                good_remaining += 1
            elif self.key_grid[i] == self.bad_color:
                bad_remaining += 1
            elif self.key_grid[i] == "Civilian":
                neutral_remaining += 1
            elif self.key_grid[i] == "Assassin":
                assasin_remaining += 1
            else:
                # TODO should never get here, throw error
                pass

        return good_remaining == 0 or bad_remaining == 0 or assasin_remaining == 0

    def _other_color(self, color):
        return "Blue" if color == "Red" else "Red"

    def _hstack_word_vectors(self, word):
        """For word, stack all word embedding nd.array for each kind of word vector"""
        try:
            stacked_words = self.all_vectors[0][word]
            for vec in self.all_vectors[1:]:
                stacked_words = np.hstack((stacked_words, vec[word]))
            return stacked_words
        except KeyError:
            return None

    # By @Greenstick https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()