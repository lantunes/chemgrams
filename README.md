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
6-gram language model (with modified Kneser-Ney smoothing without pruning)
of DeepSMILES strings was used, created from the same corpus that the
ChemTS model was trained on. The language model informs a search using
MCTS with PUCT.

### Chemgrams vs. ChemTS

Comparison against ChemTS (each value represents the maximum _J_ score, over 10 trials):

| Method              | 2h           | 4h           | 6h           |  8h         |  Molecules/Min    |
|---------------------|:------------:|:------------:|:------------:|:-----------:|------------------:|
| ChemTS              | 4.91 ± 0.38  | 5.41 ± 0.51  | 5.49 ± 0.44  | 5.58 ± 0.50 | 40.89 ± 1.57      |
| Chemgrams+MCTS      | 10.52 ± 0.66 | 11.49 ± 0.39 | 12.44 ± 0.49 | -           | 5,948.89 ± 149.69 |
| ChemTS - Only RNN   | 4.51 ± 0.27  | 4.62 ± 0.26  | 4.79 ± 0.25  | 4.79 ± 0.25 | 41.33 ± 1.42      |
| Chemgrams           | 4.83 ± 0.34  | 4.95 ± 0.32  | 5.17 ± 0.33  | 5.17 ± 0.33 | 4,713.86 ± 72.98  |

### Chemgrams vs. GB-GA

Comparison against GB-GA:

| Method         | average *J(m)* | No. molecules | CPU time   |
|----------------|:--------------:|:-------------:|-----------:|
| GB-GA(50%)     | 6.8 ± 0.7      | 1,000         | 30 seconds |
| GB-GA(1%)      | 7.4 ± 0.9      | 1,000         | 30 seconds |
| Chemgrams+MCTS | 2.89 ± 0.23    | ~3,000        | 30 seconds |
| Chemgrams+MCTS | 8.12 ± 0.49    | ~200,000      | 30 minutes |
| Chemgrams+MCTS | 12.44 ± 0.49   | ~2.1 million  | 6 hours    |

Example of a molecule generated by Chemgrams (*J* = 12.47):
<img src="https://raw.githubusercontent.com/lantunes/chemgrams/master/assets/generated_example.png" width="100%"/>

### SMILES vs. DeepSMILES

The table below summarizes the results of comparing an N-gram language
model based on SMILES syntax to an N-gram language model based on DeepSMILES
syntax. A total of 100,000 attempts were made to generate a molecule using
each model. The percentage of generated strings which represented valid
molecules is presented in the table, as well as the percentage of unique
generated molecules (since the same molecule can be generated multiple
times), and the best _J_ score achieved.

| Syntax     | LM order | % valid | % unique | Best _J_ |
|------------|----------|---------|----------|----------|
| SMILES     | 6        |  7.22   |  4.23    | 3.46     |
| SMILES     | 10       | 17.78   | 14.99    | 4.30     |
| DeepSMILES | 6        | 21.45   | 16.39    | 4.09     |
| DeepSMILES | 10       | 55.62   | 47.50    | 5.56     |

The analysis indicates that using the DeepSMILES syntax, rather than the
SMILES syntax, results in a greater chance that a valid molecule will be
generated, and that a greater fraction of the generated valid molecules
will be unique. An interesting consequence is that a higher score is
attained, likely because more valid and unique molecules are generated
in a given number of attempts.

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

NOTE: To have KenLM support language models with the default maximum order of up to 6,
replace the cmake step above with:
```
cmake ..
```
and replace the pip install step above with:
```
pip install https://github.com/kpu/kenlm/archive/master.zip
```


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
