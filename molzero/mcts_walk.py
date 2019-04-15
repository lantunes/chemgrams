from chemgrams import *
from chemgrams.jscorer import JScorer
from molzero import *
import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

vocab = get_vocab()
min_len = 24

sa_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_sa_scores.txt'))
logp_values = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_logp_values.txt'))
cycle_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_cycle_scores.txt'))
jscorer = JScorer.init(sa_scores, logp_values, cycle_scores)

def eval_function(text):
    generated = ''.join(text)

    try:
        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)
    except Exception:
        return -1.0

    jscore = jscorer.score(sanitized)
    score = jscore / (1 + np.abs(jscore))

    print("valid: ", sanitized)
    return score

mcts = BasicMCTS(vocab, 25, eval_function)

mcts.search(['<s>'], num_simulations=100000)

best = mcts.get_best_sequence()
decoded = DeepSMILESLanguageModelUtils.decode(''.join(best[0]), start='<s>', end='</s>')
smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
print("best: %s (%s)" % (smiles, str(best[1])))
