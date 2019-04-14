import unittest
from chemgrams import NLTKDeepSMILESLanguageModel
import os
import pickle

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestNLTKDeepSMILESLanguageModel(unittest.TestCase):

    def test_tokenize(self):
        lm = NLTKDeepSMILESLanguageModel([])

        tokens = lm.tokenize("<M>CC%(113)OC</M>")
        self.assertEqual(['<M>', 'C', 'C', '%(113)', 'O', 'C', '</M>'], tokens)

        tokens = lm.tokenize("COccccN)ccnc6cc%10OC")
        self.assertEqual(['C', 'O', 'c', 'c', 'c', 'c', 'N', ')', 'c', 'c', 'n', 'c', '6', 'c', 'c', '%10', 'O', 'C'], tokens)

    def test_generate(self):
        lm = self._get_lm()

        generated = lm.generate(num_chars=15, text_seed="C", random_seed=1)
        self.assertEqual("/C=C/C=C/C=C/C=", generated)

    def test_perplexity(self):
        lm = self._get_lm()

        line = "ClCC%(113)OC"
        self.assertAlmostEqual(112.757, lm.perplexity(line), places=3)

    def test_entropy(self):
        lm = self._get_lm()

        line = "[O-]C=O)COcncccc6cns5"
        self.assertAlmostEqual(3.040, lm.entropy(line), places=3)

    def test_vocab(self):
        lm = self._get_lm()
        print(lm.vocab())
        self.assertEqual(['<M>', '[O-]', 'C', '=', 'O', ')', '[C@]', '5', 'c', '6', '</M>', 'N', '[S@@]', '[S@]', '[nH]',
                          '/', '\\', '[C@@H]', '[NH2+]', 'o', '[NH+]', 'n', '%10', '[N+]', '9', '[C@H]', '7', '[NH3+]',
                          'Cl', 'Br', '%13', '#', 'F', '[C@@]', 's', 'S', '3', '[nH+]', '[n+]', '%11', '[P@]', 'P', 'I',
                          '8', '4', '[N-]', '%12', '%14', '[n-]', '[S-]', '%16', '[O+]', '[P@@]', '[o+]', '%17', '[PH+]',
                          '%18', '%22', '[S@+]', '[S@@+]', '%15', '%28', '%24', '%19', '[S+]', '[I]', '[P+]', '%23',
                          '[s+]', '%25', '%20', '%21', '[NH-]', '[P@@H+]', '[OH+]', '[CH-]', '[P@H]', '[P@H+]',
                          '<UNK>'], lm.vocab())

        self.assertEqual(['<M>', '[O-]', 'C', '=', 'O', ')', '[C@]', '5', 'c', '6', '</M>', 'N', '[S@@]', '[S@]', '[nH]',
                          '/', '\\', '[C@@H]', '[NH2+]', 'o', '[NH+]', 'n', '%10', '[N+]', '9', '[C@H]', '7', '[NH3+]',
                          'Cl', 'Br', '%13', '#', 'F', '[C@@]', 's', 'S', '3', '[nH+]', '[n+]', '%11', '[P@]', 'P', 'I',
                          '8', '4', '[N-]', '%12', '%14', '[n-]', '[S-]', '%16', '[O+]', '[P@@]', '[o+]', '%17', '[PH+]',
                          '%18', '%22', '[S@+]', '[S@@+]', '%15', '%28', '%24', '%19', '[S+]', '[I]', '[P+]', '%23',
                          '[s+]', '%25', '%20', '%21', '[NH-]', '[P@@H+]', '[OH+]', '[CH-]', '[P@H]', '[P@H+]'],
                         lm.vocab(with_unk=False))

    def _get_lm(self):
        """
        Returns a trigram NLTKDeepSMILESLanguageModel trained on ~74,000 DeepSMILES strings.
        :return: a NLTKDeepSMILESLanguageModel
        """
        with open(os.path.join(THIS_DIR, 'resources', 'lm.pkl'), 'rb') as pickle_in:
            lm = pickle.load(pickle_in)
        return lm
