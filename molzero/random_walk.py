from chemgrams import *
from molzero import *

vocab = get_vocab()
min_len = 24

for _ in range(10000):
    state = ['<s>']

    for i in range(24):
        state += [np.random.choice(vocab)]

    generated = ''.join(state)

    try:
        decoded = DeepSMILESLanguageModelUtils.decode(generated, start='<s>', end='</s>')
        sanitized = DeepSMILESLanguageModelUtils.sanitize(decoded)
        if len(sanitized) > min_len:
            print("valid: ", sanitized)
    except Exception:
        pass
