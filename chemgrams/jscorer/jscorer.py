import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.Crippen import MolLogP

import chemgrams.sascorer as sascorer


class JScorer:
    """
    Provides a score, J, for a given molecule, S:
     J(S) = logP(S) − SA(S) − RingPenalty(S)
    """
    def __init__(self, sa_mean, sa_std, logp_mean, logp_std, cycle_mean, cycle_std):
        self._sa_mean = sa_mean
        self._sa_std = sa_std
        self._logp_mean = logp_mean
        self._logp_std = logp_std
        self._cycle_mean = cycle_mean
        self._cycle_std = cycle_std

    @staticmethod
    def init(sa_scores, logp_values, cycle_scores):
        return JScorer(np.mean(sa_scores), np.std(sa_scores), np.mean(logp_values), np.std(logp_values),
                       np.mean(cycle_scores), np.std(cycle_scores))

    def score(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        try:
            logp = MolLogP(mol)
        except:
            logp = -1000

        sa_score = -sascorer.calculateScore(mol)
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        cycle_score = -cycle_length
        sa_score_norm = (sa_score - self._sa_mean) / self._sa_std
        logp_norm = (logp - self._logp_mean) / self._logp_std
        cycle_score_norm = (cycle_score - self._cycle_mean) / self._cycle_std

        return sa_score_norm + logp_norm + cycle_score_norm
