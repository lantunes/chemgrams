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
        return self.score_mol(mol)

    def score_mol(self, mol):
        mol_fp = AllChem.GetMorganFingerprint(mol, self._radius)
        return DataStructs.DiceSimilarity(self._target_fp, mol_fp)

    @staticmethod
    def score_mols(mol1, mol2, radius=4):
        mol1_fp = AllChem.GetMorganFingerprint(mol1, radius)
        mol2_fp = AllChem.GetMorganFingerprint(mol2, radius)
        return DataStructs.DiceSimilarity(mol1_fp, mol2_fp)
