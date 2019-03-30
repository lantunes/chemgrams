import pickle
from rdkit import Chem
import deepsmiles


class DeepSMILESLanguageModelUtils:

    @staticmethod
    def get_lm(file_path):
        with open(file_path, 'rb') as pickle_in:
            lm = pickle.load(pickle_in)
        return lm

    @staticmethod
    def sanitize(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise Exception("could not convert SMILES to RDKit Mol: %s" % smi)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    @staticmethod
    def extract(generated_raw):
        generated_raw = generated_raw.replace('<M>', '', 1)  # remove the leading <M>, if present
        generated_raw = generated_raw.split("</M>")[0]  # keep only the text generated up until the </M> character
        return generated_raw

    @staticmethod
    def decode(generated):
        generated = DeepSMILESLanguageModelUtils.extract(generated)
        converter = deepsmiles.Converter(rings=True, branches=True)
        return converter.decode(generated)
