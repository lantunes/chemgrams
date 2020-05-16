from rdkit import Chem
import deepsmiles


class DeepSMILESLanguageModelUtils:

    @staticmethod
    def sanitize(smi):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            raise Exception("could not convert SMILES to RDKit Mol: %s" % smi)
        # if we don't exclude SANITIZE_FINDRADICALS, then we get [C] in about 10% of generated mols
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
        # we somehow still get SMILES that cannot be later converted to a Mol, if we just call Chem.MolToSmiles, and
        #  not follow it with a call to Chem.CanonSmiles
        return Chem.CanonSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))

    @staticmethod
    def extract(generated_raw, start='<M>', end='</M>'):
        generated_raw = generated_raw.replace(start, '', 1)  # remove the leading <M>, if present
        generated_raw = generated_raw.split(end)[0]  # keep only the text generated up until the </M> character
        return generated_raw

    @staticmethod
    def decode(generated, start='<M>', end='</M>'):
        generated = DeepSMILESLanguageModelUtils.extract(generated, start, end)
        converter = deepsmiles.Converter(rings=True, branches=True)
        return converter.decode(generated)


class SMILESLanguageModelUtils:

    @staticmethod
    def sanitize(smi):
        return DeepSMILESLanguageModelUtils.sanitize(smi)

    @staticmethod
    def extract(generated_raw, start='<M>', end='</M>'):
        return DeepSMILESLanguageModelUtils.extract(generated_raw, start, end)