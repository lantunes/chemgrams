from rdkit.Chem.Crippen import MolLogP
from rdkit.RDLogger import logger

from chemgrams import *
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

logger = logger()

vocab = get_arpa_vocab('../resources/chemts_250k_deepsmiles_klm_6gram_190414.arpa')
lm = EmptyDeepSMILESLanguageModel(vocab, n=6)

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score < current_best_score

for i in range(1000):
    generated = lm.generate(num_chars=25, text_seed="<s>")
    try:

        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        mol = Chem.MolFromSmiles(sanitized)
        logp_score = MolLogP(mol)

        logger.info("successful: %s , score: %s" % (sanitized, str(logp_score)))

        if current_best_score is None or beats_current(logp_score):
            current_best_score = logp_score
            current_best_smiles = sanitized

    except Exception as e:
        pass

logger.info("best: %s , score: %s" % (current_best_smiles, str(current_best_score)))
