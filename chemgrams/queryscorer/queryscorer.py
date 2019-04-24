from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


class QueryScorer:
    """
    Based on the similarity-guided structure generation score described in:
     "Molecular de-novo design through deep reinforcement learning", Olivecrona et al. J Cheminform (2017) 9:48

    Returns a score between -1.0 and 1.0.
    """
    def __init__(self, target_smiles, radius=2, k=0.7):
        self._k = k
        self._target_mol = Chem.MolFromSmiles(target_smiles)
        self._radius = radius
        self._target_fp = AllChem.GetMorganFingerprint(self._target_mol, self._radius, useCounts=True, useFeatures=True)

    def score(self, smiles):
        mol = Chem.MolFromSmiles(smiles)

        fp = AllChem.GetMorganFingerprint(mol, 2, useCounts=True, useFeatures=True)
        score = DataStructs.TanimotoSimilarity(self._target_fp, fp)
        score = min(score, self._k) / self._k
        return -1.0 + (2*float(score))
