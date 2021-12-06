from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import scipy.spatial.distance
import itertools
import heapq
from typing import Tuple, List

from players.codemaster import *

class VectorCodemaster(Codemaster):
    """Generalized Vector Codemaster
    Concat any keyed vector that can be accessed like: word_vector_dict["<word>"] = nd.array
    """
    def __init__(self, **kwargs):
        """Set up word list and handle pretrained vectors"""
        super().__init__()

        # word vectors
        glove_vecs = kwargs.get("glove_vecs", None)
        word_vectors = kwargs.get("word_vectors", None)

        # color choice
        self.color = kwargs.get("color")
        if self.color == "Red":
            self.bad_color = "Blue"
        elif self.color == "Blue":
            self.bad_color = "Red"

        # set up word vectors
        self.all_vectors = []
        if glove_vecs is not None:
            self.all_vectors.append(glove_vecs)
        if word_vectors is not None:
            self.all_vectors.append(word_vectors)
        if "vectors" in kwargs:
            for vecs in kwargs["vectors"]:
                self.all_vectors.append(vecs)

        self.distance_threshold = kwargs.get("distance_threshold", 0.7)
        self.max_good_words_per_clue = kwargs.get(f"max_{self.color}_words_per_clue", 3)
        self.same_clue_patience = kwargs.get("sameCluePatience", 25)

        # some other word things
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer()
        self.word_distances = None
        self.words_on_board = None
        self.key_grid = None
        self.good_color = kwargs['color']
        self.bad_color = self._other_color(self.good_color)
        self.max_clue_num = 3
        self.max_depth = 2

        # set beam size
        self.beam_size = 4

        # build word set
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

    # Conduct a beam search on the game tree
    # returns the ideal clue along with the number of guesses
    def beam(self, color, words_on_board, current_heuristic) -> Tuple[str, int, int]:
        # if this is a terminal state, end recursion
        if current_heuristic == -10 or current_heuristic == 10:
            return (None, 0, current_heuristic)

        # store the best choices as (word, num, heuristic, words_on_board) 4-tuples
        best_choices = []
        for potentialClue in self.cm_word_set:
            # get the heuristic for each clue and for each min clue count
            for num in range(1, self.max_clue_num + 1):
                # simulate guesses / generate the successor state
                new_words_on_board = self._simulate_guesses(color, potentialClue, num, words_on_board)
                new_heuristic = self._heuristicFunction(color, new_words_on_board)
                # add possibility to list of best choices
                best_choices.append((potentialClue, num, new_heuristic, new_words_on_board))
        best_choices = sorted(best_choices, key=lambda x: x[2], reverse=True)
        best_choices = best_choices[0:self.beam_size]
        # store best deeper choices as (word, num, final_heuristic) 3-tuples
        beam_results = []
        new_color = None
        # change color
        if color == self.good_color:
            new_color = self.bad_color
        else:
            new_color = self.good_color
        # for size of beam
        for i in range(self.beam_size):
            current_choice = best_choices[i]
            _, number, final_heuristic = self.beam(new_color, current_choice[3], current_choice[2])
            beam_results.append((current_choice[0], current_choice[1], final_heuristic))
        beam_results = sorted(beam_results, key=lambda x: x[2], reverse=True)
        return beam_results[0]


    def get_clue(self) -> Tuple[str, int]:
        """Function that returns a clue word and number of estimated related words on the board"""
        clue, num, final_heuristic = self.beam(self.good_color, self.words_on_board,
                                 self._heuristicFunction(self.good_color, self.words_on_board))
        print(f"The {self.good_color} clue is {clue} {num}")
        print("Old board + Expected new board")
        print(self.words_on_board)
        new_board = self._simulate_guesses(self.good_color, clue, num, self.words_on_board)
        print(new_board)
        return (clue, num)

    # generate the next state
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


    # heuristic function to pick best options
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
            return 10

        if bad_remaining == 0:
            return -10

        if assasin_remaining == 0 and last_player_color == self.good_color:
            return -10
        elif assasin_remaining == 0 and last_player_color == self.bad_color:
            return 10

        return bad_remaining - good_remaining