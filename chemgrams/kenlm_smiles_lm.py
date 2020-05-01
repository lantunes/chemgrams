import kenlm
import re

import numpy as np

from .smiles_tokenizer import SMILESTokenizer
from .language_model import ChemgramsLanguageModel


class KenLMSMILESLanguageModel(ChemgramsLanguageModel):
    def __init__(self, kenlm_bin_path, vocab):
        self._lm = kenlm.Model(kenlm_bin_path)
        self._vocab = vocab
        self._n = self._lm.order
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
                    tokens.extend(self._tokenize_smiles(s))
        return tokens

    def _tokenize_smiles(self, smiles):
        tokenizer = SMILESTokenizer(smiles)
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
        tokenized = self._tokenize_context(context)
        if tokenized is None:
            tokenized = []
        sequence = tokenized + [char]
        bos = True
        eos = True

        if len(sequence) > 0 and sequence[0] == '<s>':
            sequence = sequence[1:]
        else:
            bos = False

        if len(sequence) > 0 and sequence[-1] == '</s>':
            sequence = sequence[:-1]
        else:
            eos = False

        score = self._lm.score(' '.join(sequence), bos=bos, eos=eos)
        return 10**score  # KenLM scores are log10(prob)

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
        """
        Do not include <s> or </s>.
        :param text: the text to evaluate, without <s> and </s>
        :return: the perplexity
        """
        sequence = self._tokenize_context(text)
        if len(sequence) > 0 and sequence[0] == '<s>':
            sequence = sequence[1:]
        if len(sequence) > 0 and sequence[-1] == '</s>':
            sequence = sequence[:-1]
        return self._lm.perplexity(' '.join(sequence))

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
        """
        Returns the log10 probability of a sentence. A sentence should be tokenized and in KenLM format,
        e.g. "C C ( C ) O C"
        :param sentence:
        :return: returns the log base 10 probability of a sentence
        """
        return self._lm.score(sentence)
