from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


class TanimotoScorer:
    def __init__(self, target_smiles):
        self._target_mol = Chem.MolFromSmiles(target_smiles)
        self._target_fp = AllChem.GetMorganFingerprint(self._target_mol, 4)  # roughly equivalent to the ECFP4 fingerprint

    def score(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol_fp = AllChem.GetMorganFingerprint(mol, 4)
        return DataStructs.DiceSimilarity(self._target_fp, mol_fp)
