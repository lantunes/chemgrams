from rdkit import Chem
import deepsmiles
import selfies as sf


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

    @staticmethod
    def extract_sentence(generated_tokens, start='<s>', end='</s>', join_on=''):
        """
        Takes a list of generated tokens, that may or may not contain start and end tokens,
        and returns a string, joined on join_on, excluding any start and end tokens.
        :param generated_tokens: a list of tokens; e.g. ['<s>', 'C', 'C', 'O', '</s>']
        :param start: the beginning-of-sentence token
        :param end: the end-of-sentence token
        :param join_on: the string on which to join the tokens
        :return: a string, joined on the join string, excluding the start and end tokens
                 e.g. if join_on=' ' and generated_tokens=['<s>', 'C', 'C', 'O', '</s>'], then 'C C O' will be returned
        """
        sequence = list(generated_tokens)

        if len(sequence) > 0 and sequence[0] == start:
            sequence = sequence[1:]

        if len(sequence) > 0 and sequence[-1] == end:
            sequence = sequence[:-1]

        return join_on.join(sequence)


class SMILESLanguageModelUtils:

    @staticmethod
    def sanitize(smi):
        return DeepSMILESLanguageModelUtils.sanitize(smi)

    @staticmethod
    def extract(generated_raw, start='<M>', end='</M>'):
        return DeepSMILESLanguageModelUtils.extract(generated_raw, start, end)


class SELFIESLanguageModelUtils:

    @staticmethod
    def sanitize(smi):
        return DeepSMILESLanguageModelUtils.sanitize(smi)

    @staticmethod
    def extract(generated_raw, start='<M>', end='</M>'):
        return DeepSMILESLanguageModelUtils.extract(generated_raw, start, end)

    @staticmethod
    def decode(generated, start='<M>', end='</M>'):
        generated = SELFIESLanguageModelUtils.extract(generated, start, end)
        return sf.decoder(generated)
