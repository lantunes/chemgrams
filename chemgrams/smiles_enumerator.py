from rdkit import Chem
import numpy as np


class SMILESEnumerator:
    def __init__(self):
        pass

    @staticmethod
    def randomize_smiles(smiles, n=1, isomericSmiles=True):
        """
        Creates randomized versions of the given SMILES. Up to n SMILES strings will be returned.
        :param smiles: the SMILES string to be randomized
        :param n: the maximum number of randomized SMILES to generate
        :param isomericSmiles: whether to use isomeric SMILES
        :return: a set of unique randomized SMILES strings (which may contain less than n members if n > 1)
        """
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        randomized = set()
        for _ in range(n):
            np.random.shuffle(ans)
            nm = Chem.RenumberAtoms(m, ans)
            randomized.add(Chem.MolToSmiles(nm, canonical=False, isomericSmiles=isomericSmiles))
        return randomized
