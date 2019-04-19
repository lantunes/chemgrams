import os
import time
from chemgrams import *
from chemgrams.jscorer import JScorer
from rdkit.RDLogger import logger
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = logger()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    logger.info("loading language model...")

    vocab = get_arpa_vocab('../models/chemts_250k_deepsmiles_klm_6gram_190414.arpa')
    lm = KenLMDeepSMILESLanguageModel('../models/chemts_250k_deepsmiles_klm_6gram_190414.klm', vocab)

    num_simulations = 1500000
    width = 12
    max_depth = 100
    start_state = ["<s>"]
    c = 2

    sa_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_sa_scores.txt'))
    logp_values = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_logp_values.txt'))
    cycle_scores = np.loadtxt(os.path.join(THIS_DIR, '..', 'resources', 'chemts_cycle_scores.txt'))
    jscorer = JScorer.init(sa_scores, logp_values, cycle_scores)

    def eval_function(text):
        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            return -1.0

        jscore = jscorer.score(smiles)
        score = jscore / (1 + np.abs(jscore))

        logger.info("%s, %s" % (generated, str(score)))
        return score

    mcts = LanguageModelMCTSWithPUCTTerminating(lm, width, max_depth, eval_function, cpuct=c, terminating_symbol='</s>')
    state = start_state

    logger.info("beginning search...")
    start = time.time()
    mcts.search(state, num_simulations)
    end = time.time()

    best = mcts.get_best_sequence()
    generated_text = ''.join(best[0])
    logger.info("generated text: %s" % generated_text)
    decoded = DeepSMILESLanguageModelUtils.decode(generated_text, start='<s>', end='</s>')
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
    logger.info("SMILES: %s, J: %s (%s seconds)" % (smiles, jscorer.score(smiles), str((end - start))))
