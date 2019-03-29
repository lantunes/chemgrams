import re
import pybel
from nltk.lm import Laplace
from nltk.lm.preprocessing import flatten
from nltk.util import everygrams
from deepsmiles import Converter

from .deepsmiles_tokenizer import DeepSMILESTokenizer


class DeepSMILESLanguageModel:
    def __init__(self, smiles_corpus, n=3):
        self._converter = Converter(rings=True, branches=True)
        tokens = []
        for smiles in smiles_corpus:
            deepsmiles = self._smiles_to_deepsmiles(smiles)
            line_tokens = self._tokenize_deepsmiles(deepsmiles)
            line_tokens.insert(0, "<M>")
            line_tokens.append("</M>")
            tokens.append(line_tokens)
        t = (everygrams(x, max_len=n) for x in tokens)
        v = flatten(tokens)
        lm = Laplace(order=n)  # add-one smoothing
        lm.fit(t, v)

        self._n = n
        self._lm = lm
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

    def _smiles_to_deepsmiles(self, smiles):
        canonical = pybel.readstring("smi", smiles).write("can").strip()
        return self._converter.encode(canonical)

    def _tokenize_deepsmiles(self, deepsmiles):
        tokenizer = DeepSMILESTokenizer(deepsmiles)
        tokens = tokenizer.get_tokens()
        return [t.value for t in tokens]

    def _tokenize_context(self, context=None):
        if context is None or type(context) == str:
            tokenized = self.tokenize(context)
        elif type(context) == list:
            tokenized = context
        else:
            raise Exception("unsupported context type: ", type(context))
        return tokenized

    def generate(self, num_chars=1, text_seed=None, random_seed=None):
        tokenized = self._tokenize_context(text_seed)
        generated = self._lm.generate(num_words=num_chars, text_seed=tokenized, random_seed=random_seed)
        return ''.join(generated)

    def perplexity(self, text):
        tokenized = self._tokenize_context(text)
        train = (everygrams(tokenized, max_len=self._n))
        return self._lm.perplexity(train)

    def entropy(self, text):
        tokenized = self._tokenize_context(text)
        train = (everygrams(tokenized, max_len=self._n))
        return self._lm.entropy(train)

    def vocab(self, with_unk=True):
        if not with_unk:
            return [w for w in self._lm.vocab if w != "<UNK>"]
        return [w for w in self._lm.vocab]

    def score(self, char, context=None):
        tokenized = self._tokenize_context(context)
        return self._lm.score(char, tokenized)

    def vocab_scores(self, context=None):
        all = []
        tokenized = self._tokenize_context(context)
        for v in self._lm.vocab:
            all.append((v, self._lm.score(v, tokenized)))
        return reversed(sorted(all, key=lambda k: k[1]))

    def top_n_vocab(self, n, context=None):
        top_n = []
        vocab_scores = self.vocab_scores(context)
        for i in range(n):
            v = next(vocab_scores)
            if v[0] == "<UNK>":
                v = next(vocab_scores)
            top_n.append(v[0])
        return top_n

    def top_n_vocab_with_weights(self, n, context=None):
        top_n = ([], [])
        vocab_scores = self.vocab_scores(context)
        for i in range(n):
            v = next(vocab_scores)
            if v[0] == "<UNK>":
                v = next(vocab_scores)
            top_n[0].append(v[0])
            top_n[1].append(v[1])
        return top_n[0], self._normalize(top_n[1])

    def _normalize(self, probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]

    def order(self):
        return self._n
