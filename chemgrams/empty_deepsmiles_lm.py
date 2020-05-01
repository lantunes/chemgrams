import re

import numpy as np

from .deepsmiles_tokenizer import DeepSMILESTokenizer
from .language_model import ChemgramsLanguageModel


class EmptyDeepSMILESLanguageModel(ChemgramsLanguageModel):
    def __init__(self, vocab, n=3):
        self._vocab = vocab
        self._n = n
        self._tokenize_pattern = re.compile(r'(<s>)|(</s>)')

    def tokenize(self, text):
        if text is None: return None
        split = self._tokenize_pattern.split(text)
        tokens = []
        for s in split:
            if s != '' and s is not None:
                if s == "<s>" or s == "</s>":
                    tokens.append(s)
                else:
                    tokens.extend(self._tokenize_deepsmiles(s))
        return tokens

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

    def score(self, char, context=None):
        return np.random.uniform(0, 1)

    def vocab(self, with_unk=True):
        if not with_unk:
            return [w for w in self._vocab if w != "<unk>"]
        return [w for w in self._vocab]

    def vocab_scores(self, context=None):
        all = []
        tokenized = self._tokenize_context(context)
        for v in self._vocab:
            all.append((v, self.score(v, tokenized)))
        return reversed(sorted(all, key=lambda k: k[1]))

    def top_n_vocab(self, n, context=None):
        top_n = []
        vocab_scores = self.vocab_scores(context)
        for i in range(n):
            v = next(vocab_scores)
            if v[0] == "<unk>":
                v = next(vocab_scores)
            top_n.append(v[0])
        return top_n

    def top_n_vocab_with_weights(self, n, context=None):
        top_n = ([], [])
        vocab_scores = self.vocab_scores(context)
        for i in range(n):
            v = next(vocab_scores)
            if v[0] == "<unk>":
                v = next(vocab_scores)
            top_n[0].append(v[0])
            top_n[1].append(v[1])
        return top_n[0], self._normalize(top_n[1])

    def _normalize(self, probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]

    def perplexity(self, text):
        return 1.0

    def generate(self, num_chars=1, text_seed=None, random_seed=None):
        context = self._tokenize_context(text_seed)
        if num_chars == 0:
            return context
        if context is None or len(context) == 0:
            context = ['<s>']
            num_chars -= 1
        while num_chars > 0 and context[-1] != '</s>':
            context += [self._select_next_randomly(context)]
            num_chars -= 1
        return ''.join(context)

    def _select_next_randomly(self, context):
        top_n_tokens = self.top_n_vocab_with_weights(len(self._vocab), context[-self.order() + 1:])
        return np.random.choice(top_n_tokens[0], p=top_n_tokens[1])

    def order(self):
        return self._n

    def log_prob(self, sentence):
        return 0.0
