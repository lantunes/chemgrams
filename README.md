Chemgrams
=========

N-gram language models of DeepSMILES strings, combined with MCTS.

Comparison against ChemTS (https://arxiv.org/abs/1710.00616):

| Method              | 2h           | 4h           | 6h           |  8h         |  Molecules/Min    |
|---------------------|:------------:|:------------:|:------------:|:-----------:|------------------:|
| ChemTS              | 4.91 ± 0.38  | 5.41 ± 0.51  | 5.49 ± 0.44  | 5.58 ± 0.50 | 40.89 ± 1.57      |
| Chemgrams           | 10.52 ± 0.66 | 11.49 ± 0.39 | 12.44 ± 0.49 | -           | 5,948.89 ± 149.69 |
| ChemTS - Only RNN   | 4.51 ± 0.27  | 4.62 ± 0.26  | 4.79 ± 0.25  | 4.79 ± 0.25 | 41.33 ± 1.42      |
| Chemgrams - Only LM | 4.83 ± 0.34  | 4.95 ± 0.32  | 5.17 ± 0.33  | 5.17 ± 0.33 | 4,713.86 ± 72.98  |

Comparison against GB-GA (https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c#!divAbstract):

| Method      | average *J(m)* | No. molecules | CPU time   |
|-------------|:--------------:|:-------------:|-----------:|
| GB-GA(50%)  | 6.8 ± 0.7      | 1,000         | 30 seconds |
| GB-GA(1%)   | 7.4 ± 0.9      | 1,000         | 30 seconds |
| Chemgrams   | 2.89 ± 0.23    | ~3,000        | 30 seconds |
| Chemgrams   | 8.12 ± 0.49    | ~200,000      | 30 minutes |
| Chemgrams   | 12.44 ± 0.49   | ~2.1 million  | 6 hours    |


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
