import deepsmiles

from chemgrams import DeepSMILESLanguageModelUtils


converter = deepsmiles.Converter(rings=True, branches=True)

lm = DeepSMILESLanguageModelUtils.get_lm("models/chembl_25_deepsmiles_lm_5gram_190330.pkl")

generated = lm.generate(num_chars=15, text_seed="<M>")

print(generated)

generated = generated.split("</M>")[0]  # keep only the text generated up until the termination character

print(generated)

decoded = converter.decode(generated)

sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)

print(sanitized)
