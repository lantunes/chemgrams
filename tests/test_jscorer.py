import os
import unittest

import numpy as np

from chemgrams.jscorer import JScorer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestJScorer(unittest.TestCase):

    def test_score(self):
        sa_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_sa_scores.txt'))
        logp_values = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_logp_values.txt'))
        cycle_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_cycle_scores.txt'))

        jscorer = JScorer.init(sa_scores, logp_values, cycle_scores)

        self.assertAlmostEqual(6.506,
                               jscorer.score("O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)c2ccccc2c1OC(F)F)c1cccc2ccccc12"),
                               places=3)

        self.assertAlmostEqual(6.381,
                               jscorer.score("O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)ccc1C1=CCCCC1)c1cc(F)cc(Cl)c1"),
                               places=3)

        self.assertAlmostEqual(5.777,
                               jscorer.score("O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)c(Cl)cc1Cl)c1cccs1"),
                               places=3)

        self.assertAlmostEqual(5.201,
                               jscorer.score("CCc1nc(N2C=CCC2)ccc1-c1c(C)cc(CF)c(-c2cccc(C(C)C)c2)c1-c1c(C)ccc(C)c1C"),
                               places=3)

        self.assertAlmostEqual(4.532,
                               jscorer.score("CCC1(c2c(-c3c(C)cc(S)cc3C)c(C)c(-c3c(C)cc(C)cc3O)c(Br)c2-c2cc(C)c(C)c(S)c2)C=COC1"),
                               places=3)
