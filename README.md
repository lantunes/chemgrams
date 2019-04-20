Chemgrams
=========

Chemgrams are N-gram language models of DeepSMILES, a SMILES-like syntax.
Chemgrams can be combined with Monte Carlo Tree Search (MCTS) to search
chemical space for molecules with desired properties. Chemgrams also
refers to this Python software library.

Chemgrams has been compared to existing, state-of-the-art methods for
generating novel molecules with desired properties, such as ChemTS (https://arxiv.org/abs/1710.00616), and
GB-GA (https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c#!divAbstract).

The manuscript describing Chemgrams is currently in preparation, and will
be submitted to a journal for peer review. A link to the paper will be
posted here as soon as it becomes available.

Comparison against ChemTS:

| Method              | 2h           | 4h           | 6h           |  8h         |  Molecules/Min    |
|---------------------|:------------:|:------------:|:------------:|:-----------:|------------------:|
| ChemTS              | 4.91 ± 0.38  | 5.41 ± 0.51  | 5.49 ± 0.44  | 5.58 ± 0.50 | 40.89 ± 1.57      |
| Chemgrams+MCTS      | 10.52 ± 0.66 | 11.49 ± 0.39 | 12.44 ± 0.49 | -           | 5,948.89 ± 149.69 |
| ChemTS - Only RNN   | 4.51 ± 0.27  | 4.62 ± 0.26  | 4.79 ± 0.25  | 4.79 ± 0.25 | 41.33 ± 1.42      |
| Chemgrams           | 4.83 ± 0.34  | 4.95 ± 0.32  | 5.17 ± 0.33  | 5.17 ± 0.33 | 4,713.86 ± 72.98  |

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

$ cmake ..

$ make -j 4

# if not in chemgrams_env
$ source activate chemgrams_env

$ pip install https://github.com/kpu/kenlm/archive/master.zip

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
