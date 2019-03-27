import re
from .deepsmiles_tokenizer import DeepSMILESTokenizer


class DeepSMILESLanguageModel:
    def __init__(self, n=3):
        self._n = n
        self._tokenize_pattern = re.compile(r'(<M>)|(</M>)')

    def tokenize(self, text):
        if text is None: return None
        split = self._tokenize_pattern.split(text)
        tokens = []
        for s in split:
            if s != '' and s is not None:
                if s == "<M>" or s == "</M>":
                    tokens.append(s)
                else:
                    tokens.extend(self._tokenize_deepsmiles(s))
        return tokens

    def _tokenize_deepsmiles(self, deepsmiles):
        tokenizer = DeepSMILESTokenizer(deepsmiles)
        tokens = tokenizer.get_tokens()
        return [t.value for t in tokens]
