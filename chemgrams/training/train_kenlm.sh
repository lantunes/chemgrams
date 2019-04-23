#!/bin/sh
# This script assumes the kenlm bin folder is on the PATH; the bin folder is usually found in the kenlm build folder.
# This script expects 3 arguments:
#  1. the location of the corpus text file (e.g. "models/chemts_250k_deepsmiles_corpus_kenlm.txt")
#  2. the destination directory for the output (e.g. "models")
#  3. a name to use for the generated .arpa and .klm files (e.g. "chemts_250k_deepsmiles_klm_6gram_190414")

lmplz -o 6 --discount_fallback --text $1 --arpa $2/$3.arpa

build_binary $2/$3.arpa $2/$3.klm
