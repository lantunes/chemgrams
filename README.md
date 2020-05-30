Chemgrams
=========

Chemgrams are N-gram language models of [DeepSMILES](https://chemrxiv.org/articles/DeepSMILES_An_Adaptation_of_SMILES_for_Use_in_Machine-Learning_of_Chemical_Structures/7097960/1) _(N. O'Boyle, 2018)_, a SMILES-like syntax.
Chemgrams can be combined with Monte Carlo Tree Search (MCTS) to search
chemical space for molecules with desired properties. Chemgrams also
refers to this Python software library. The [KenLM](https://kheafield.com/code/kenlm/)
toolkit is used for rapid language model estimation and sampling. Chemgrams
aspires to be a lightweight alternative to RNN-based chemical language models.

Chemgrams has been compared to existing, state-of-the-art methods for
generating novel molecules with desired properties, such as [ChemTS](https://arxiv.org/abs/1710.00616) _(K. Tsuda et al., 2017)_, and
[GB-GA](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c#!divAbstract) _(J. H. Jensen, 2019)_.

_The manuscript describing Chemgrams is currently in preparation, and will
be submitted to a journal for peer review. A link to the paper will be
posted here as soon as it becomes available._

In the tables below, the values represent a _J_ score, defined as:
```
J(S) = logP(S) − SA(S) − RingPenalty(S)
```
for a molecule _S_, and _SA_ is the synthetic accessibility score. Also, a
6- or 10-gram language model (with modified Kneser-Ney smoothing without pruning)
of DeepSMILES strings was used, estimated from the same corpus that the
ChemTS model was trained on (~250,000 SMILES strings). The language model
informs a search using MCTS with PUCT.

### Chemgrams vs. ChemTS

Comparison against ChemTS (each value represents the maximum _J_ score, over 10 trials):

| Method               | 2h           | 4h           | 6h           |  8h         |  Valid Molecules/min. |
|----------------------|:------------:|:------------:|:------------:|:-----------:|----------------------:|
| ChemTS               | 4.91 ± 0.38  | 5.41 ± 0.51  | 5.49 ± 0.44  | 5.58 ± 0.50 | 40.89 ± 1.57          |
| Chemgrams(n=6)+MCTS  | 10.52 ± 0.66 | 11.49 ± 0.39 | 12.44 ± 0.49 | N/A *       | 2,581.85 ± 60.99      |
| Chemgrams(n=10)+MCTS | 10.90 ± 0.54 | 12.55 ± 0.56 | 13.36 ± 0.74 | N/A *       | 3,226.45 ± 111.69     |
| ChemTS - Only RNN    | 4.51 ± 0.27  | 4.62 ± 0.26  | 4.79 ± 0.25  | 4.79 ± 0.25 | 41.33 ± 1.42          |
| Chemgrams(n=6)       | 4.83 ± 0.34  | 4.95 ± 0.32  | 5.17 ± 0.33  | 5.17 ± 0.33 | 1,016.44 ± 16.10      |
| Chemgrams(n=10)      | 6.18 ± 0.43  | 6.55 ± 0.39  | 6.74 ± 0.45  | 6.75 ± 0.43 | 2,137.12 ± 21.70      |

_* due to memory limitations, and the large number of molecules generated, search was halted earlier_

### Chemgrams vs. GB-GA

Comparison against GB-GA:

| Method              | Average *J(m)* | No. Molecules | CPU time   |
|---------------------|:--------------:|:-------------:|-----------:|
| GB-GA(50%)          | 6.8 ± 0.7      | 1,000         | 30 seconds |
| GB-GA(1%)           | 7.4 ± 0.9      | 1,000         | 30 seconds |
| Chemgrams(n=6)+MCTS | 2.89 ± 0.23    | ~3,000        | 30 seconds |
| Chemgrams(n=6)+MCTS | 8.12 ± 0.49    | ~200,000      | 30 minutes |
| Chemgrams(n=6)+MCTS | 12.44 ± 0.49   | ~2.1 million  | 6 hours    |

Example of a molecule generated by Chemgrams (*J* = 12.47):
<img src="https://raw.githubusercontent.com/lantunes/chemgrams/master/assets/generated_example.png" width="100%"/>

### SMILES vs. DeepSMILES

The table below summarizes the results of comparing an N-gram language
model based on SMILES syntax to an N-gram language model based on DeepSMILES
syntax. A total of 100,000 attempts were made to generate a molecule using
each model. The percentage of generated strings which represented valid
molecules is presented in the table, as well as the percentage of the
generated valid molecules which are unique (since the same molecule can
be generated multiple times), and the best _J_ score achieved.

| Syntax     | LM Order | % Valid | % Unique | Best _J_ |
|------------|:--------:|:-------:|:--------:|:--------:|
| SMILES     | 6        |  7.22   | 58.59    | 3.46     |
| SMILES     | 10       | 17.78   | 84.31    | 4.30     |
| DeepSMILES | 6        | 21.45   | 76.41    | 4.09     |
| DeepSMILES | 10       | 55.62   | 85.40    | 5.56     |

The analysis indicates that using the DeepSMILES syntax, rather than the
SMILES syntax, results in a greater chance that a valid molecule will be
generated, and that a greater fraction of the generated valid molecules
will be unique. An interesting consequence is that a higher score is
attained, likely because more valid and unique molecules are generated
in a given number of attempts.

The perplexities of the models were computed for a test corpus of
~50,000 held out SMILES strings:

| Syntax     | LM Order | Perplexity |
|------------|:--------:|:----------:|
| SMILES     | 6        |  9.429     |
| SMILES     | 10       |  8.443     |
| DeepSMILES | 6        |  3.748 *   |
| DeepSMILES | 10       |  3.637 *   |

_* the SMILES string was first converted to DeepSMILES before scoring_

### Generated Corpus Characteristics

A number of attempts were made to generate a molecule using either
the 10-gram DeepSMILES Language Model, or MCTS informed by the 10-gram
DeepSMILES Language Model, estimated from the ChemTS corpus (which consists
of ~250,000 molecules). The results are presented in the following table:

| Method                | # Generated | # Valid          | # Unique         | # Seen in Training     | Time Required |
|:---------------------:|:-----------:|-----------------:|-----------------:|:----------------------:|:-------------:|
| Chemgrams(n=10)       | 500,000     | 278,638 (55.73%) | 215,087 (43.02%) | 419 (0.19% of Unique)  |  ~122 minutes |
| Chemgrams(n=10)+MCTS* | 500,000     | 283,537 (56.71%) | 261,903 (53.38%) |  22 (0.008% of Unique) |   ~87 minutes |
| Chemgrams(n=10)+MCTS^ | 500,000     | 242,440 (48.49%) | 228,922 (45.78%) |  82 (0.036% of Unique) |  ~103 minutes |

_* a reward of 1.0 was given if the molecule was valid, and -1.0 if it was either invalid or already generated_

_^ a reward of -1.0 was given of the molecules was invalid or already generated, otherwise the score was: log(p_prior(s)) + σ,
  where σ is a tuning parameter, set to 2, and log(p_prior(s)) is the log probability of the generated sequence according to
  the language model_

When MCTS is used, the uniqueness of the generated molecules increases,
and much fewer of the molecules seen in training are generated. However,
the distribution of molecules generated using LM-informed MCTS, as
characterized by seven physico-chemical descriptors, has drifted away
from the characteristics of the molecules of the original corpus used to
create the LM. This can be visualized with the following t-SNE plots:

<img src="https://raw.githubusercontent.com/lantunes/chemgrams/master/assets/kenlm_deepsmiles_10gram_tsne.png" width="60%"/>

The image above is a t-SNE plot for a sampling of the molecules generated
using the LM alone, described by seven physico-chemical properties. There
is strong overlap between the original and generated corpora.

<img src="https://raw.githubusercontent.com/lantunes/chemgrams/master/assets/kenlm_mcts_deepsmiles_10gram_tsne.png" width="60%"/>

The image above is a t-SNE plot for a sampling of the molecules generated
using the LM and MCTS. This plot demonstrates that the character of
the generated molecules has drifted away from that of the corpus used to
create the LM.

To ameliorate this drift, the score returned during MCTS can be augmented
with the log probability of the generated sequence according to the
language model (as in [Olivecrona et al. 2017](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x)).
The resulting t-SNE plot below demonstrates that the generated molecules
lie closer to the corpus used to create the language model:

<img src="https://raw.githubusercontent.com/lantunes/chemgrams/master/assets/kenlm_mcts_deepsmiles_prior_10gram_tsne.png" width="48%"/>

### Influence of the Corpus

The syntax error rate and the fraction of unique molecules also depends
on the corpus used to estimate the language model:

| Corpus           | Size      | % Valid | % Unique | Best _J_ |
|------------------|:---------:|:-------:|:--------:|:--------:|
| Zinc12 Fragments | 5,015     | 70.67   | 49.94    | 2.66     |
| ChemTS Corpus    | 249,456   | 55.62   | 85.40    | 5.56     |
| ChEMBL Corpus    | 1,765,191 | 44.92   | 85.99    | 10.97    |

In the table above, DeepSMILES syntax was used, along with a 10-gram
language model. A total of 100,000 attempts were made to generate a
molecule using each model. It should be noted that the larger corpus,
while producing a higher error rate, produced molecules with a much
better score. This is likely a reflection of the variety of molecules
that can be generated by the model.

### Influence of Canonicalization

One way to augment or enhance a SMILES corpus is to enumerate the SMILES
strings, so that a number of non-canonical versions of the string are
produced for each original string, and added to the corpus. This has been termed
[SMILES Enumeration](https://arxiv.org/abs/1703.07076). It was
[reported](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0393-0)
to increase the quality of molecular generative models. This section summarizes
the results of applying SMILES Enumeration to corpora used to create
n-gram language models.

Some of the corpora in the following table have undergone SMILES Enumeration. That is,
each molecule was randomized in terms of heavy atom numbering, and the resulting SMILES
was added to the corpus; this was done 10 times for each molecule in
the corpus:

| Syntax     | Corpus           | Enumerated? | % Valid | % Unique | Best _J_ |
|------------|:----------------:|:-----------:|:-------:|:--------:|:--------:|
| SMILES     | ChemTS Corpus    | No          | 17.78   | 84.31    | 4.30     |
| SMILES     | ChemTS Corpus    | Yes         |  8.49   | 80.70    | 4.11     |
| DeepSMILES | ChemTS Corpus    | No          | 55.62   | 85.40    | 5.56     |
| DeepSMILES | ChemTS Corpus    | Yes         | 11.28   | 91.71    | 3.95     |
| DeepSMILES | Zinc12 Fragments | No          | 70.67   | 49.94    | 2.66     |
| DeepSMILES | Zinc12 Fragments | Yes         | 59.88   | 57.80    | 2.66     |

_NOTE: the enumerated corpora are ~10-fold larger than the non-enumerated corpora_

It appears that enhancing the corpora with SMILES enumerations of the
original molecules degrades the ability of the model to produce valid
syntax, though it may lead to more uniqueness in the generated set. This
seems counter-intuitive. After all, the resulting corpora are ~10-fold
larger. Shouldn't that improve the ability of the model to generate valid
syntax? One reason for this behaviour may be that the SMILES strings in
the corpora are canonicalized, and some consistency between various
structural motifs exists in the corpus, in terms of how the SMILES are
written. By adding various SMILES representations of the same molecule,
that consistency is lost, and it somewhat "confuses" the model, as there
is no consistent way to represent certain structural motifs with
SMILES (or DeepSMILES) syntax.

The following models use non-enumerated, standard corpora, but with
various canonicalization strategies:

| Syntax     | Corpus           | Canonicalizer    | % Valid | % Unique | Best _J_ |
|------------|:----------------:|:----------------:|:-------:|:--------:|:--------:|
| DeepSMILES | Zinc12 Fragments | pybel            | 70.67   | 49.94    | 2.66     |
| DeepSMILES | Zinc12 Fragments | rdkit            | 68.39   | 49.80    | 2.66     |
| DeepSMILES | Zinc12 Fragments | none             | 70.00   | 47.71    | 2.53     |
| DeepSMILES | Zinc12 Fragments | de-canonicalized | 49.35   | 59.19    | 2.66     |

It appears that the corpus is already canonicalized to an extent, as when we perform
no canonicalization, and just use the SMILES strings as they appear in the original
corpus, there is no change in the outcome. But when we "de-canonicalize" the corpus
(by producing up to 50 enumerations of each original SMILES in the corpus, and randomly
choosing one of the enumerations in place of the original SMILES), we see an increase
in the syntax error rate (and an increase in the uniqueness, as we saw for the
enumerated corpora).

So it appears that the language model quality is sensitive to whether the SMILES strings
in the corpus are consistently canonicalized or not. Increasing the size of the corpus
by adding enumerations of the canonicalized SMILES strings will result in poorer
model quality. To ensure that the best model is produced, the SMILES strings of the
corpus must be consistently canonicalized.

## Setup On OSX

1. Download Open Babel from its original source (https://github.com/openbabel/openbabel)

2. Extract it and cd into it (openbabel-master)

3. Make a directory “build” and cd into it.

4. Run the following commands:

```
$ conda create -n chemgrams_env python=3.6

$ source activate chemgrams_env

$ pip install deepsmiles

$ pip install networkx

$ pip install numpy

$ which python3
/Users/me/anaconda/envs/chemgrams_env/bin/python3

# if cmake is not installed
$ brew install cmake

$ cmake ../ -DPYTHON_EXECUTABLE=/Users/me/anaconda/envs/chemgrams_env/bin/python3 -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON

$ make

$ make install

$ conda install -c openbabel openbabel

$ conda install -c conda-forge rdkit
```

5. Install KenLM (on OSX):

```
$ brew install cmake boost eigen

$ git clone https://github.com/kpu/kenlm.git

$ cd kenlm

$ mkdir -p build && cd build

$ cmake -DKENLM_MAX_ORDER=10 ..

$ make -j 4

# if not in chemgrams_env
$ source activate chemgrams_env

$ pip install https://github.com/kpu/kenlm/archive/master.zip --install-option="--max_order=10"

```

_NOTE: The settings above create a KenLM installation where the default maximum order of a language model is 10._

### Docker Image

To build the Docker image:
```
$ docker build -t chemgrams .
```

To start a container:
```
$ docker run --name chemgrams_bash --rm chemgrams
```

To stop the container:
```
$ docker stop chemgrams
```
