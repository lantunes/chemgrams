import os
import time
import numpy as np

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils, \
    LanguageModelMCTSWithPUCTTerminating
from chemgrams.qedscorer import QEDScorer

from rdkit.RDLogger import logger
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = logger()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    logger.info("loading language model...")

    vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_6gram_190414.arpa')
    lm = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_6gram_190414.klm', vocab)

    num_simulations = 100000
    width = 24
    max_depth = 100
    start_state = ["<s>"]
    c = 5

    qedscorer = QEDScorer()

    all_smiles = {}

    def eval_function(text):
        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            return -1.0

        global all_smiles

        if smiles in all_smiles:
            score = -1.0
        else:
            qedscore = qedscorer.score(smiles)
            score = qedscore / (1 + np.abs(qedscore))
            all_smiles[smiles] = qedscore

        logger.info("%s, %s" % (smiles, str(score)))
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
    logger.info("SMILES: %s, J: %s (%s seconds)" % (smiles, qedscorer.score(smiles), str((end - start))))

    all_best = reversed(list(reversed(sorted(all_smiles.items(), key=lambda kv: kv[1])))[:5])
    for i, ab in enumerate(all_best):
        print("%d. %s, %s" % (5-i, ab[0], str(ab[1])))
