from rdkit.RDLogger import logger

from chemgrams import *
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = logger()


if __name__ == '__main__':

    logger.info("loading language model...")
    vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_10gram_200429.arpa')
    lm = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)

    num_simulations = 1000
    width = 3
    text_length = 25
    start_state = ["<s>"]

    def eval_function(text):
        generated = ''.join(text)
        try:
            decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
            DeepSMILESLanguageModelUtils.sanitize(decoded)
        except Exception:
            return 0
        extracted = DeepSMILESLanguageModelUtils.extract(generated, start='<s>', end='</s>')
        tokenized = DeepSMILESTokenizer(extracted)
        len_reward = len(tokenized.get_tokens()) / (text_length - 1)  # provide more reward for longer text sequences
        perplexity = lm.perplexity(text)
        perplexity_reward = perplexity / (1 + perplexity)
        score = (perplexity_reward*0.5) + (len_reward*0.5)

        logger.info("%s, %s" % (generated, str(score)))
        return score

    mcts = LanguageModelMCTSWithUCB1(lm, width, text_length, eval_function)
    state = start_state

    logger.info("beginning search...")
    mcts.search(state, num_simulations)

    best = mcts.get_best_sequence()

    generated_text = ''.join(best[0])
    logger.info("generated text: %s (score: %s, perplexity: %s)" % (generated_text, str(best[1]), lm.perplexity(generated_text)))

    decoded = DeepSMILESLanguageModelUtils.decode(generated_text, start='<s>', end='</s>')
    smiles = DeepSMILESLanguageModelUtils.sanitize(decoded)

    logger.info("SMILES: %s" % smiles)
