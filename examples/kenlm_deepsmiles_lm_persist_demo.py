import os
import time

from chemgrams import get_arpa_vocab, KenLMDeepSMILESLanguageModel, DeepSMILESLanguageModelUtils
from chemgrams.logger import get_logger

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = get_logger('chemgrams.log')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

logger.info("LM-only")
logger.info("KenLMDeepSMILESLanguageModel(n=10, 'chemts_250k_deepsmiles_klm_10gram_200429.klm')")
logger.info("num_chars=100, text_seed='<s>'")


vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_10gram_200429.arpa')
lm = KenLMDeepSMILESLanguageModel('../resources/chemts_250k_deepsmiles_klm_10gram_200429.klm', vocab)

all_smiles = set()
num_valid = 0

start = time.time()
for i in range(500000):  # about enough to get ~250,000 valid molecules
    try:
        generated = lm.generate(num_chars=100, text_seed='<s>')

        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        num_valid += 1

        all_smiles.add(sanitized)

    except Exception as e:
        pass

    if (i+1) % 5000 == 0:
        logger.info("--iteration: %d--" % (i+1))
        logger.info("num valid: %d" % num_valid)
        logger.info("num unique: %s" % len(all_smiles))

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
