import unittest
import deepsmiles as ds
import pybel
from chemgrams import DeepSMILESTokenizer, DeepSMILESToken
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDeepSMILESTokenizer(unittest.TestCase):

    def test_it(self):
        smi = "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)c2ccccc2c1OC(F)F)c1cccc2ccccc12"
        converter = ds.Converter(rings=True, branches=True)
        canonical = pybel.readstring("smi", smi).write("can").strip()
        deepsmiles = converter.encode(canonical)
        tokenizer = DeepSMILESTokenizer(deepsmiles)
        print(len(tokenizer.get_tokens()))

    def test_can_tokenize_deepsmiles(self):
        with open(os.path.join(THIS_DIR, 'resources', 'smiles.txt'), 'r') as content_file:
            lines = content_file.readlines()
        all_smiles = [line.strip() for line in lines]

        converter = ds.Converter(rings=True, branches=True)
        for smiles in all_smiles:
            canonical = pybel.readstring("smi", smiles).write("can").strip()
            deepsmiles = converter.encode(canonical)
            DeepSMILESTokenizer(deepsmiles)

    def test_molecule1(self):
        tokenizer = DeepSMILESTokenizer("[O-]C=O)COcncccc6cns5")

        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], DeepSMILESToken("formal_atom", "[O-]", 0))
        self.assertEqual(tokens[1], DeepSMILESToken("plain_atom", "C", 4))
        self.assertEqual(tokens[2], DeepSMILESToken("bond", "=", 5))
        self.assertEqual(tokens[3], DeepSMILESToken("plain_atom", "O", 6))
        self.assertEqual(tokens[4], DeepSMILESToken("grouping", ")", 7))
        self.assertEqual(tokens[5], DeepSMILESToken("plain_atom", "C", 8))
        self.assertEqual(tokens[6], DeepSMILESToken("plain_atom", "O", 9))
        self.assertEqual(tokens[7], DeepSMILESToken("plain_atom", "c", 10))
        self.assertEqual(tokens[8], DeepSMILESToken("plain_atom", "n", 11))
        self.assertEqual(tokens[9], DeepSMILESToken("plain_atom", "c", 12))
        self.assertEqual(tokens[10], DeepSMILESToken("plain_atom", "c", 13))
        self.assertEqual(tokens[11], DeepSMILESToken("plain_atom", "c", 14))
        self.assertEqual(tokens[12], DeepSMILESToken("plain_atom", "c", 15))
        self.assertEqual(tokens[13], DeepSMILESToken("ring_size", "6", 16))
        self.assertEqual(tokens[14], DeepSMILESToken("plain_atom", "c", 17))
        self.assertEqual(tokens[15], DeepSMILESToken("plain_atom", "n", 18))
        self.assertEqual(tokens[16], DeepSMILESToken("plain_atom", "s", 19))
        self.assertEqual(tokens[17], DeepSMILESToken("ring_size", "5", 20))

        self.assertEqual(tokenizer.get_raw_length(), 21)
        self.assertEqual(tokenizer.get_tokenized_length(), 18)

        t = tokenizer.token_at(3)
        self.assertEqual(t, DeepSMILESToken("plain_atom", "O", 6))

    def test_molecule2(self):
        tokenizer = DeepSMILESTokenizer("COccccN)ccnc6cc%10OC")

        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], DeepSMILESToken("plain_atom", "C", 0))
        self.assertEqual(tokens[1], DeepSMILESToken("plain_atom", "O", 1))
        self.assertEqual(tokens[2], DeepSMILESToken("plain_atom", "c", 2))
        self.assertEqual(tokens[3], DeepSMILESToken("plain_atom", "c", 3))
        self.assertEqual(tokens[4], DeepSMILESToken("plain_atom", "c", 4))
        self.assertEqual(tokens[5], DeepSMILESToken("plain_atom", "c", 5))
        self.assertEqual(tokens[6], DeepSMILESToken("plain_atom", "N", 6))
        self.assertEqual(tokens[7], DeepSMILESToken("grouping", ")", 7))
        self.assertEqual(tokens[8], DeepSMILESToken("plain_atom", "c", 8))
        self.assertEqual(tokens[9], DeepSMILESToken("plain_atom", "c", 9))
        self.assertEqual(tokens[10], DeepSMILESToken("plain_atom", "n", 10))
        self.assertEqual(tokens[11], DeepSMILESToken("plain_atom", "c", 11))
        self.assertEqual(tokens[12], DeepSMILESToken("ring_size", "6", 12))
        self.assertEqual(tokens[13], DeepSMILESToken("plain_atom", "c", 13))
        self.assertEqual(tokens[14], DeepSMILESToken("plain_atom", "c", 14))
        self.assertEqual(tokens[15], DeepSMILESToken("ring_size", "%10", 15))
        self.assertEqual(tokens[16], DeepSMILESToken("plain_atom", "O", 18))
        self.assertEqual(tokens[17], DeepSMILESToken("plain_atom", "C", 19))

        self.assertEqual(tokenizer.get_raw_length(), 20)
        self.assertEqual(tokenizer.get_tokenized_length(), 18)

        t = tokenizer.token_at(9)
        self.assertEqual(t, DeepSMILESToken("plain_atom", "c", 9))

    def test_ring_size(self):
        tokenizer = DeepSMILESTokenizer("CC%(113)OC")
        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], DeepSMILESToken("plain_atom", "C", 0))
        self.assertEqual(tokens[1], DeepSMILESToken("plain_atom", "C", 1))
        self.assertEqual(tokens[2], DeepSMILESToken("ring_size", "%(113)", 2))
        self.assertEqual(tokens[3], DeepSMILESToken("plain_atom", "O", 8))
        self.assertEqual(tokens[4], DeepSMILESToken("plain_atom", "C", 9))

    def test_chlorine(self):
        tokenizer = DeepSMILESTokenizer("ClCC%(113)OC")
        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], DeepSMILESToken("plain_atom", "Cl", 0))
        self.assertEqual(tokens[1], DeepSMILESToken("plain_atom", "C", 2))
        self.assertEqual(tokens[2], DeepSMILESToken("plain_atom", "C", 3))
        self.assertEqual(tokens[3], DeepSMILESToken("ring_size", "%(113)", 4))
        self.assertEqual(tokens[4], DeepSMILESToken("plain_atom", "O", 10))
        self.assertEqual(tokens[5], DeepSMILESToken("plain_atom", "C", 11))

    def test_bromine(self):
        tokenizer = DeepSMILESTokenizer("BrCC%(113)OC")
        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], DeepSMILESToken("plain_atom", "Br", 0))
        self.assertEqual(tokens[1], DeepSMILESToken("plain_atom", "C", 2))
        self.assertEqual(tokens[2], DeepSMILESToken("plain_atom", "C", 3))
        self.assertEqual(tokens[3], DeepSMILESToken("ring_size", "%(113)", 4))
        self.assertEqual(tokens[4], DeepSMILESToken("plain_atom", "O", 10))
        self.assertEqual(tokens[5], DeepSMILESToken("plain_atom", "C", 11))

    def test_bonds(self):
        tokenizer = DeepSMILESTokenizer("CC/C=C/C-CC#N")
        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], DeepSMILESToken("plain_atom", "C", 0))
        self.assertEqual(tokens[1], DeepSMILESToken("plain_atom", "C", 1))
        self.assertEqual(tokens[2], DeepSMILESToken("bond", "/", 2))
        self.assertEqual(tokens[3], DeepSMILESToken("plain_atom", "C", 3))
        self.assertEqual(tokens[4], DeepSMILESToken("bond", "=", 4))
        self.assertEqual(tokens[5], DeepSMILESToken("plain_atom", "C", 5))
        self.assertEqual(tokens[6], DeepSMILESToken("bond", "/", 6))
        self.assertEqual(tokens[7], DeepSMILESToken("plain_atom", "C", 7))
        self.assertEqual(tokens[8], DeepSMILESToken("bond", "-", 8))
        self.assertEqual(tokens[9], DeepSMILESToken("plain_atom", "C", 9))
        self.assertEqual(tokens[10], DeepSMILESToken("plain_atom", "C", 10))
        self.assertEqual(tokens[11], DeepSMILESToken("bond", "#", 11))
        self.assertEqual(tokens[12], DeepSMILESToken("plain_atom", "N", 12))
