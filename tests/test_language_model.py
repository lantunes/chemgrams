import unittest
from chemgrams import DeepSMILESLanguageModel


class TestDeepSMILESLanguageModel(unittest.TestCase):

    def test_tokenize(self):
        lm = DeepSMILESLanguageModel()

        tokens = lm.tokenize("<M>CC%(113)OC</M>")
        self.assertEqual(['<M>', 'C', 'C', '%(113)', 'O', 'C', '</M>'], tokens)

        tokens = lm.tokenize("COccccN)ccnc6cc%10OC")
        self.assertEqual(['C', 'O', 'c', 'c', 'c', 'c', 'N', ')', 'c', 'c', 'n', 'c', '6', 'c', 'c', '%10', 'O', 'C'], tokens)
