import random
from math import *
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms


class MolecularMCTS:
    def __init__(self, allowed_angle_values, energy_function, c=sqrt(2)):
        self._allowed_angle_values = allowed_angle_values
        self._energy_function = energy_function
        self._c = c

    def init_state(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        mol = Chem.AddHs(mol,explicitOnly=False)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        self._rotatable_bonds = self._get_rotatable_bonds(mol)
        self._num_angles = len(self._rotatable_bonds)
        return mol, []

    def _get_rotatable_bonds(self, molecule):
        raw_rot_bonds =  molecule.GetSubstructMatches(Chem.MolFromSmarts("[!#1]~[!$(*#*)&!D1]-!@[!$(*#*)&!D1]~[!#1]"))
        raw_rot_bonds += molecule.GetSubstructMatches(Chem.MolFromSmarts("[*]~[*]-[O,S]-[#1]"))
        raw_rot_bonds += molecule.GetSubstructMatches(Chem.MolFromSmarts("[*]~[*]-[NX3;H2]-[#1]"))
        bonds = []
        rot_bonds = []
        for k, i, j, l in raw_rot_bonds:
            if (i, j) not in bonds:
                bonds.append((i,j))
                rot_bonds.append((k, i, j, l))
        return rot_bonds

    def get_num_angles(self):
        if self._num_angles is None:
            raise Exception("init_state must be called first")
        return self._num_angles

    def search(self, state, num_simulations):
        """
        :param state: a tuple of (Mol object, list of processed dihedral angles) 
        :param num_simulations: the number of simulations to perform
        :return: a state representing the minimal energy conformer
        """
        root_node = _Node(state, self._rotatable_bonds, self._allowed_angle_values, self._c)

        # Perform simulations
        for i in range(num_simulations):
            node = root_node

            # Select
            while not node.has_untried_moves() and node.has_children():
                node = node.select_child()

            # Expand
            if node.has_untried_moves():
                move_state = node.select_untried_move()
                node = node.add_child(move_state, self._rotatable_bonds, self._allowed_angle_values, self._c)

            # Rollout
            rollout_state = node.state
            while len(rollout_state[1]) < self._num_angles:
                rollout_state = self._select_next_move_randomly(rollout_state)

            # Backpropagate
            #   backpropagate from the expanded node and work back to the root node
            energy = self._energy_function(rollout_state[0])
            while node is not None:
                node.visits += 1
                node.energies.append(energy)
                node = node.parent

        # return the move that was most visited
        most_visited_node = sorted(root_node.children, key = lambda c: c.visits)[-1]
        return most_visited_node.state

    def _select_next_move_randomly(self, state):
        mol = Chem.Mol(state[0])  # create a copy of the Mol
        conf = mol.GetConformer()
        bond_to_rotate = self._rotatable_bonds[len(state[1])]
        theta = rdMolTransforms.GetDihedralDeg(conf, bond_to_rotate[0], bond_to_rotate[1], bond_to_rotate[2], bond_to_rotate[3])
        theta += np.random.choice(self._allowed_angle_values)
        rdMolTransforms.SetDihedralDeg(conf, bond_to_rotate[0], bond_to_rotate[1], bond_to_rotate[2], bond_to_rotate[3], theta)
        conformer_id = mol.AddConformer(conf, assignId=True)
        return Chem.Mol(mol, False, conformer_id), state[1] + [theta]


class _Node:
    def __init__(self, state, rotatable_bonds, allowed_angle_values, c, parent=None):
        self.state = state
        self._rotatable_bonds = rotatable_bonds
        self._c = c
        self._allowed_angle_values = allowed_angle_values
        self.energies = []
        self.visits = 0
        self.parent = parent
        self.children = []
        self.untried_moves = self._get_child_states()

    def _get_child_states(self):
        child_states = []
        if len(self.state[1]) < len(self._rotatable_bonds):
            mol = self.state[0]
            conf = mol.GetConformer()
            mol.AddConformer(conf, assignId=True)
            bond_to_rotate = self._rotatable_bonds[len(self.state[1])]
            theta = rdMolTransforms.GetDihedralDeg(conf, bond_to_rotate[0], bond_to_rotate[1], bond_to_rotate[2], bond_to_rotate[3])
            for allowed_angle_value in self._allowed_angle_values:
                new_theta = theta + allowed_angle_value
                rdMolTransforms.SetDihedralDeg(conf, bond_to_rotate[0], bond_to_rotate[1], bond_to_rotate[2], bond_to_rotate[3], new_theta)
                mol.AddConformer(conf, assignId=True)
            mol.RemoveConformer(conf.GetId())

            for i, conf in enumerate(mol.GetConformers()):
                theta = rdMolTransforms.GetDihedralDeg(conf, bond_to_rotate[0], bond_to_rotate[1], bond_to_rotate[2], bond_to_rotate[3])
                child_mol = Chem.Mol(mol, False, conf.GetId())
                child_states.append((child_mol, self.state[1] + [theta]))

        return child_states

    def _average_value(self):
        merit = -(np.array(self.energies) / (1 + np.abs(self.energies)))
        return np.sum(merit) / self.visits

    def has_untried_moves(self):
        return self.untried_moves != []

    def select_untried_move(self):
        return random.choice(self.untried_moves)

    def add_child(self, child_state, rotatable_bonds, allowed_angle_values, c):
        child = _Node(child_state, rotatable_bonds, allowed_angle_values, c, parent=self)
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
