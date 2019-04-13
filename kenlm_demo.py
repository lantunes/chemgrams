from rdkit.Chem.Crippen import MolLogP

from chemgrams import *


vocab = get_arpa_vocab('models/chembl_25_deepsmiles_klm_5gram_190413.arpa')
lm = DeepSMILESKenLM('models/chembl_25_deepsmiles_klm_5gram_190413.klm', vocab)

print(lm.score('C', '<s>CCCCCCC'))
print(lm.score('c', '<s>CCCCCCC'))
print(lm.score('c', '<s>ccccc'))
print(lm.score('N', '<s>CCCCCCC'))
print(lm.score('=', '<s>CCCCCCC'))

vocab_scores = lm.vocab_scores(context='<s>')
for v in vocab_scores:
    print(v)

print('---------')

print(lm.perplexity("CCCccccCC"))
print(lm.perplexity("CCCBBB"))

print(lm.perplexity("<s>CCCBBB"))

print('---------')

top_n = lm.top_n_vocab(3, '<s>')
print(top_n)

print('---------')

top_n = lm.top_n_vocab_with_weights(3, '<s>')
print(top_n)

print('---------')

current_best_score = None
current_best_smiles = None
beats_current = lambda score: score > current_best_score

for i in range(1000):
    generated = lm.generate(num_chars=25, text_seed="<s>")
    try:
        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

        mol = Chem.MolFromSmiles(sanitized)
        logp_score = MolLogP(mol)

        print("successful: %s , score: %s" % (sanitized, str(logp_score)))

        if current_best_score is None or beats_current(logp_score):
            current_best_score = logp_score
            current_best_smiles = sanitized

    except Exception as e:
        pass

print("best: %s , score: %s" % (current_best_smiles, str(current_best_score)))
