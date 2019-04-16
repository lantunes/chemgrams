Chemgrams
=========

N-gram language models of DeepSMILES strings, combined with MCTS.


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
