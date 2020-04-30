import unittest
from chemgrams import SMILESTokenizer, SMILESToken
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestSMILESTokenizer(unittest.TestCase):

    def test_can_tokenize_smiles(self):
        with open(os.path.join(THIS_DIR, 'resources', 'smiles.txt'), 'r') as content_file:
            lines = content_file.readlines()
        all_smiles = [line.strip() for line in lines]

        for smiles in all_smiles:
            SMILESTokenizer(smiles)

    def test_molecule1(self):
        tokenizer = SMILESTokenizer("C1CCCC1")

        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], SMILESToken(type='plain_atom', value='C', index=0))
        self.assertEqual(tokens[1], SMILESToken(type='ring_closure', value='1', index=1))
        self.assertEqual(tokens[2], SMILESToken(type='plain_atom', value='C', index=2))
        self.assertEqual(tokens[3], SMILESToken(type='plain_atom', value='C', index=3))
        self.assertEqual(tokens[4], SMILESToken(type='plain_atom', value='C', index=4))
        self.assertEqual(tokens[5], SMILESToken(type='plain_atom', value='C', index=5))
        self.assertEqual(tokens[6], SMILESToken(type='ring_closure', value='1', index=6))

        self.assertEqual(tokenizer.get_raw_length(), 7)
        self.assertEqual(tokenizer.get_tokenized_length(), 7)

        t = tokenizer.token_at(3)
        self.assertEqual(t, SMILESToken("plain_atom", "C", 3))

    def test_molecule2(self):
        tokenizer = SMILESTokenizer("B(c1ccccc1)(O)O")

        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], SMILESToken(type='plain_atom', value='B', index=0))
        self.assertEqual(tokens[1], SMILESToken(type='grouping', value='(', index=1))
        self.assertEqual(tokens[2], SMILESToken(type='plain_atom', value='c', index=2))
        self.assertEqual(tokens[3], SMILESToken(type='ring_closure', value='1', index=3))
        self.assertEqual(tokens[4], SMILESToken(type='plain_atom', value='c', index=4))
        self.assertEqual(tokens[5], SMILESToken(type='plain_atom', value='c', index=5))
        self.assertEqual(tokens[6], SMILESToken(type='plain_atom', value='c', index=6))
        self.assertEqual(tokens[7], SMILESToken(type='plain_atom', value='c', index=7))
        self.assertEqual(tokens[8], SMILESToken(type='plain_atom', value='c', index=8))
        self.assertEqual(tokens[9], SMILESToken(type='ring_closure', value='1', index=9))
        self.assertEqual(tokens[10], SMILESToken(type='grouping', value=')', index=10))
        self.assertEqual(tokens[11], SMILESToken(type='grouping', value='(', index=11))
        self.assertEqual(tokens[12], SMILESToken(type='plain_atom', value='O', index=12))
        self.assertEqual(tokens[13], SMILESToken(type='grouping', value=')', index=13))
        self.assertEqual(tokens[14], SMILESToken(type='plain_atom', value='O', index=14))

        self.assertEqual(tokenizer.get_raw_length(), 15)
        self.assertEqual(tokenizer.get_tokenized_length(), 15)

        t = tokenizer.token_at(8)
        self.assertEqual(t, SMILESToken("plain_atom", "c", 8))

    def test_ring_size(self):
        tokenizer = SMILESTokenizer("CC%(113)OC")
        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], SMILESToken("plain_atom", "C", 0))
        self.assertEqual(tokens[1], SMILESToken("plain_atom", "C", 1))
        self.assertEqual(tokens[2], SMILESToken("ring_closure", "%(113)", 2))
        self.assertEqual(tokens[3], SMILESToken("plain_atom", "O", 8))
        self.assertEqual(tokens[4], SMILESToken("plain_atom", "C", 9))

    def test_chlorine(self):
        tokenizer = SMILESTokenizer("ClCC%(113)OC")
        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], SMILESToken("plain_atom", "Cl", 0))
        self.assertEqual(tokens[1], SMILESToken("plain_atom", "C", 2))
        self.assertEqual(tokens[2], SMILESToken("plain_atom", "C", 3))
        self.assertEqual(tokens[3], SMILESToken("ring_closure", "%(113)", 4))
        self.assertEqual(tokens[4], SMILESToken("plain_atom", "O", 10))
        self.assertEqual(tokens[5], SMILESToken("plain_atom", "C", 11))

    def test_bromine(self):
        tokenizer = SMILESTokenizer("[BH]CC%(113)OC")
        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], SMILESToken("formal_atom", "[BH]", 0))
        self.assertEqual(tokens[1], SMILESToken("plain_atom", "C", 4))
        self.assertEqual(tokens[2], SMILESToken("plain_atom", "C", 5))
        self.assertEqual(tokens[3], SMILESToken("ring_closure", "%(113)", 6))
        self.assertEqual(tokens[4], SMILESToken("plain_atom", "O", 12))
        self.assertEqual(tokens[5], SMILESToken("plain_atom", "C", 13))

    def test_bonds(self):
        tokenizer = SMILESTokenizer("CC/C=C/C-CC#N")
        tokens = tokenizer.get_tokens()
        self.assertEqual(tokens[0], SMILESToken("plain_atom", "C", 0))
        self.assertEqual(tokens[1], SMILESToken("plain_atom", "C", 1))
        self.assertEqual(tokens[2], SMILESToken("bond", "/", 2))
        self.assertEqual(tokens[3], SMILESToken("plain_atom", "C", 3))
        self.assertEqual(tokens[4], SMILESToken("bond", "=", 4))
        self.assertEqual(tokens[5], SMILESToken("plain_atom", "C", 5))
        self.assertEqual(tokens[6], SMILESToken("bond", "/", 6))
        self.assertEqual(tokens[7], SMILESToken("plain_atom", "C", 7))
        self.assertEqual(tokens[8], SMILESToken("bond", "-", 8))
        self.assertEqual(tokens[9], SMILESToken("plain_atom", "C", 9))
        self.assertEqual(tokens[10], SMILESToken("plain_atom", "C", 10))
        self.assertEqual(tokens[11], SMILESToken("bond", "#", 11))
        self.assertEqual(tokens[12], SMILESToken("plain_atom", "N", 12))
