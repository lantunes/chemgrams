import os
import time
from threading import Timer

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils, \
    LanguageModelMCTSWithPUCTTerminating
from chemgrams.logger import get_logger

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    logger.info(os.path.basename(__file__))
    logger.info("KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)")
    logger.info("width = 24, max_depth = 100, start_state = ['<s>'], c = 5")
    logger.info("score: -1.0 if invalid; -1.0 if seen previously; 1.0 if valid")
    logger.info("ret_score = prior_log_prob + sigma*score; sigma = 2")
    logger.info("LanguageModelMCTSWithPUCTTerminating")

    LOG_INTERVAL = 60.0

    logger.info("loading language model...")

    vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_10gram_200429.arpa')
    prior = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)

    num_simulations = 500000  # about enough to get ~250,000 valid molecules

    width = 24
    max_depth = 100
    start_state = ["<s>"]
    c = 5
    sigma = 2

    all_smiles = set()
    num_valid = 0
    i = 0


    def log_progress():
        global t
        logger.info("--results--")
        logger.info("num valid: %d" % num_valid)
        logger.info("num unique: %s" % len(all_smiles))
        logger.info("num iterations: %s" % i)
        t = Timer(LOG_INTERVAL, log_progress)
        t.start()
    t = Timer(LOG_INTERVAL, log_progress)
    t.start()


    def eval_function(text):
        global i, num_valid, all_smiles

        i += 1

        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            return -1.0

        num_valid += 1

        if smiles in all_smiles:
            return -1.0
        else:
            # the score in this case is simply 1.0, since the molecule is valid and hasn't been generated yet;
            #  but it could be anything, such as whether the generated sequence contains sulfur, etc.
            score = 1.0
            all_smiles.add(smiles)

        # As in "Molecular de-novo design through deep reinforcement learning", by Olivecrona et al., we are adding
        #  the prior's log probability of the generated sequence to the score.
        prior_log_prob = prior.log_prob(DeepSMILESLanguageModelUtils.extract_sentence(text, join_on=' ', start='<s>', end='</s>'))

        tot_score = prior_log_prob + sigma*score

        # rescale the score
        # in practice, the log probs are rarely less than -45; so the min tot_score can be: -45 + (sigma*-1.0)
        rescale_min = -45 - sigma
        if tot_score < rescale_min:
            logger.info("WARNING: total score lower than %s" % rescale_min)
        # because probabilities are in the range [0,1], the max log prob is log(1) i.e. 0
        #  so the max tot_score can be: 0 + sigma*1.0
        rescale_max = sigma
        # scaling x into [a,b]: (b-a)*((x - min(x))/(max(x) - min(x))+a
        ret_score = (1 - (-1)) * ((tot_score - rescale_min)/(rescale_max - rescale_min)) + (-1)

        return ret_score

    mcts = LanguageModelMCTSWithPUCTTerminating(prior, width, max_depth, eval_function, cpuct=c, terminating_symbol='</s>')
    state = start_state

    logger.info("beginning search...")
    start = time.time()

    mcts.search(state, num_simulations)

    t.cancel()
    end = time.time()

    logger.info("--done--")
    logger.info("num valid: %d" % num_valid)
    logger.info("num unique: %s" % len(all_smiles))
    logger.info("seconds: %s" % str(end - start))

    logger.info("writing generated SMILES...")
    name = "./%s-generated.txt" % os.path.basename(__file__).split(".")[0]
    with open(name, 'w') as f:
        for smi in all_smiles:
            f.write(smi)
            f.write("\n")
