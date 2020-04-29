import random
from math import *
import math
import numpy as np


class TorsionalMCTS:
    def __init__(self, num_angles, allowed_angle_values, energy_function, c=sqrt(2)):
        self._num_angles = num_angles
        self._allowed_angle_values = allowed_angle_values
        self._energy_function = energy_function
        self._c = c

    def search(self, state, num_simulations):
        root_node = _Node(state, self._num_angles, self._allowed_angle_values, self._c)

        # Perform simulations
        for i in range(num_simulations):
            node = root_node

            # Select
            while not node.has_untried_moves() and node.has_children():
                node = node.select_child()

            # Expand
            if node.has_untried_moves():
                move_state = node.select_untried_move()
                node = node.add_child(move_state, self._num_angles, self._allowed_angle_values, self._c)

            # Rollout
            rollout_state = list(node.state)
            while len(rollout_state) < self._num_angles:
                rollout_state += [self._select_next_move_randomly()]

            # Backpropagate
            #   backpropagate from the expanded node and work back to the root node
            energy = self._energy_function(rollout_state)
            while node is not None:
                node.visits += 1
                node.energies.append(energy)
                node = node.parent

        # return the move that was most visited
        most_visited_node = sorted(root_node.children, key = lambda c: c.visits)[-1]
        return most_visited_node.state

    def _select_next_move_randomly(self):
        return np.random.choice(self._allowed_angle_values)


class _Node:
    def __init__(self, state, num_angles, allowed_angle_values, c, parent=None):
        self.state = state
        self._c = c
        self._num_angles = num_angles
        self._allowed_angle_values = allowed_angle_values
        self.energies = []
        self.visits = 0
        self.parent = parent
        self.children = []
        self.untried_moves = self._get_child_states()

    def _get_child_states(self):
        child_states = []
        for allowed_angle_value in self._allowed_angle_values:
            child_states.append(self.state + [allowed_angle_value])
        return child_states

    def _average_value(self):
        return -np.mean(self.energies)

    def has_untried_moves(self):
        return self.untried_moves != []

    def select_untried_move(self):
        return random.choice(self.untried_moves)

    def add_child(self, child_state, num_angles, allowed_angle_values, c):
        child = _Node(child_state, num_angles, allowed_angle_values, c, parent=self)
        self.children.append(child)
        self.untried_moves.remove(child_state)
        return child

    def has_children(self):
        return self.children != []

    def select_child(self):
        highest_ucb1 = None
        selected_child_node = None
        for child_node in self.children:
            ucb1 = child_node.ucb1()
            if highest_ucb1 is None or highest_ucb1 < ucb1:
                highest_ucb1 = ucb1
                selected_child_node = child_node
        return selected_child_node

    def ucb1(self):
        if self.visits == 0:
            return math.inf
        return self._average_value() + self._c*sqrt(log(self.parent.visits)/self.visits)