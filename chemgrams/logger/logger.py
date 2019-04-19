import logging


def get_logger(filename=None, name='chemgrams'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if filename is not None:
        fh = logging.FileHandler(filename, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_top_best(pairs, top_n, logger):
    """
    pairs should be a dictionary, with key=SMILES, value=(score, generated DeepSMILES)
    """
    all_best = reversed(list(reversed(sorted(pairs.items(), key=lambda kv: kv[1][0])))[:top_n])
    for i, ab in enumerate(all_best):
        logger.info("%d. %s, %s" % (top_n - i, ab[0], str(ab[1])))
