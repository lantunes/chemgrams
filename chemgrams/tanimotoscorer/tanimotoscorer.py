from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


class TanimotoScorer:
    def __init__(self, target_smiles, radius=4):
        self._target_mol = Chem.MolFromSmiles(target_smiles)
        # a radius of 2 is roughly equivalent to the ECFP4 fingerprint
        self._radius = radius
        self._target_fp = AllChem.GetMorganFingerprint(self._target_mol, self._radius)

    def score(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol_fp = AllChem.GetMorganFingerprint(mol, self._radius)
        return DataStructs.DiceSimilarity(self._target_fp, mol_fp)
