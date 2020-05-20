import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops


class CycleScorer:
    """
    A scorer that penalizes unrealistically large rings. For example, a molecule with an 8-membered ring
    would receive a score of 2, a molecule with a 7-membered ring would receive a score of 1, and a molecule with
    zero or more 6-membered or less rings would receive a score of 0.

    In general: score = 0 if largest_ring_size <= 6, else largest_ring_size - 6
    """
    def __init__(self):
        pass

    def score(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return self.score_mol(mol)

    def score_mol(self, mol):
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        return cycle_length