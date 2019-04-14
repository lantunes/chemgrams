import random
from math import *
import math
import numpy as np


class LanguageModelMCTSWithPUCTTerminating:
    """
    The tree expansion stops at nodes with states ending with a terminating symbol (such as '</s>').
    """
    def __init__(self, language_model, width, max_depth, eval_function, cpuct, terminating_symbol):
        self._lm = language_model
        self._width = width
        self._max_depth = max_depth
        self._eval_function = eval_function
        self._best_sequence = None
        self._cpuct = cpuct
        self._terminating_symbol = terminating_symbol

    def search(self, state, num_simulations):
        root_node = _Node(state, self._lm, self._width, self._max_depth, self._cpuct, self._terminating_symbol)

        # Perform simulations
        for i in range(num_simulations):
            node = root_node

            # Select
            while not node.has_untried_moves() and node.has_children():
                node = node.select_child()

            # Expand
            if node.has_untried_moves():
                move_state = node.select_untried_move()
                node = node.add_child(move_state, self._lm, self._width, self._max_depth, self._cpuct, self._terminating_symbol)

            # Rollout
            rollout_state = list(node.state)
            while len(rollout_state) < self._max_depth and rollout_state[-1] != self._terminating_symbol:
                rollout_state += [self._select_next_move_randomly(rollout_state, self._lm, self._width)]

            # Backpropagate
            #   backpropagate from the expanded node and work back to the root node
            score = self._eval_function(rollout_state)
            while node is not None:
                node.visits += 1
                node.wins += score
                node = node.parent

            #
            print("%s, %s" % (''.join(rollout_state), str(score)))
            #

            self._store_best(rollout_state, score)

        # return the move that was most visited
        most_visited_node = sorted(root_node.children, key = lambda c: c.visits)[-1]
        return most_visited_node.state

    def _select_next_move_randomly(self, rollout_state, language_model, width):
        top_n_tokens = language_model.top_n_vocab_with_weights(width, rollout_state[-self._lm.order() + 1:])
        return np.random.choice(top_n_tokens[0], p=top_n_tokens[1])

    def _store_best(self, rollout_state, score):
        current_best = self._best_sequence
        if current_best is None or score > current_best[1]:
            self._best_sequence = (rollout_state, score)

    def get_best_sequence(self):
        return self._best_sequence


class _Node:
    def __init__(self, state, language_model, width, max_depth, cpuct, terminating_symbol, parent=None):
        self.state = state
        self._cpuct = cpuct
        self._terminating_symbol = terminating_symbol
        self._lm = language_model
        self._width = width
        self._max_depth = max_depth
        self.wins = 0.0
        self.visits = 0.0
        self.prob = None
        self.parent = parent
        self.children = []
        self.untried_moves, self.child_weight_map = self._get_child_states()

    def _get_child_states(self):
        child_states = []
        child_state_weight_map = {}
        if len(self.state) < self._max_depth and self.state[-1] != self._terminating_symbol:
            top_n_tokens_with_weights = self._lm.top_n_vocab_with_weights(self._width, self.state[-self._lm.order() + 1:])
            for i in range(len(top_n_tokens_with_weights[0])):
                child_state = self.state + [top_n_tokens_with_weights[0][i]]
                child_states.append(child_state)
                child_state_weight_map[''.join(child_state)] = top_n_tokens_with_weights[1][i]
        return child_states, child_state_weight_map

    def _average_value(self):
        return self.wins / self.visits

    def has_untried_moves(self):
        return self.untried_moves != []

    def select_untried_move(self):
        return random.choice(self.untried_moves)

    def add_child(self, child_state, language_model, width, max_depth, c, terminating_symbol):
        child = _Node(child_state, language_model, width, max_depth, c, terminating_symbol, parent=self)
        child.prob = self.child_weight_map[''.join(child_state)]
        self.children.append(child)
        self.untried_moves.remove(child_state)
        return child

    def has_children(self):
        return self.children != []

    def select_child(self):
        highest_puct = None
        selected_child_node = None
        for child_node in self.children:
            puct = child_node.puct()
            if highest_puct is None or highest_puct < puct:
                highest_puct = puct
                selected_child_node = child_node
        return selected_child_node

    def puct(self):
        if self.visits == 0:
            return math.inf
        if self.prob is None:
            raise Exception("node has no action prob: %s" % self.state)
        return self.wins / self.visits + self._cpuct * self.prob * (sqrt(self.parent.visits) / (1 + self.visits))
