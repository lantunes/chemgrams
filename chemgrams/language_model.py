

class ChemgramsLanguageModel:

    def order(self):
        raise NotImplementedError

    def top_n_vocab_with_weights(self, n, context=None):
        raise NotImplementedError

    def perplexity(self, text):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError

    def score(self, char, context=None):
        raise NotImplementedError

    def vocab(self, with_unk=True):
        raise NotImplementedError

    def vocab_scores(self, context=None):
        raise NotImplementedError

    def top_n_vocab(self, n, context=None):
        raise NotImplementedError

    def generate(self, num_chars=1, text_seed=None, random_seed=None):
        raise NotImplementedError

    def log_prob(self, sentence):
        raise NotImplementedError
