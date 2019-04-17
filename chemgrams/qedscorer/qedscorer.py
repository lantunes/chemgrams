from rdkit import Chem
from rdkit.Chem import QED

import chemgrams.sascorer as sascorer


class QEDScorer:
    """
    Provides a score, J, for a given molecule, S:
     J(S) = 5*QED(S) − SA(S)
    as in "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules",
     Rafael Gómez-Bombarelli et al., 2017
    """
    def __init__(self):
        pass

    def score(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        qed_score = QED.qed(mol)

        sa_score = sascorer.calculateScore(mol)

        return 5*qed_score - sa_score
