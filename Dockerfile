FROM continuumio/miniconda3

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get -y autoremove && \
    apt-get clean

RUN apt-get install unzip && \
    apt-get -y install cmake && \
    apt-get install -y build-essential && \
    apt-get -y install zlib1g-dev libbz2-dev liblzma-dev && \
    apt-get install -y libboost-all-dev && \
    apt-get install -y libeigen3-dev

WORKDIR /root
RUN wget https://github.com/openbabel/openbabel/archive/openbabel-2-4-1.zip && \
    unzip openbabel-2-4-1.zip -d openbabel-2-4-1

WORKDIR /root/openbabel-2-4-1/openbabel-openbabel-2-4-1
RUN mkdir build && \
    conda create -n chemgrams_env python=3.6 && \
    echo "source activate chemgrams_env" >> ~/.bashrc && \
    /bin/bash -c "source activate chemgrams_env"

WORKDIR /root/openbabel-2-4-1/openbabel-openbabel-2-4-1/build
RUN cmake ../ -DPYTHON_EXECUTABLE=`which python3` -DPYTHON_BINDINGS=ON -DRUN_SWIG=ON && \
    make && \
    make install && \
    conda install -n chemgrams_env -c openbabel openbabel && \
    conda install -n chemgrams_env -c conda-forge rdkit

WORKDIR /root
RUN wget -O kenlm.zip https://github.com/kpu/kenlm/archive/master.zip && \
    unzip kenlm.zip -d kenlm

WORKDIR /root/kenlm/kenlm-master
RUN mkdir -p build

WORKDIR /root/kenlm/kenlm-master/build
RUN cmake -DKENLM_MAX_ORDER=10 .. && \
    make -j 4 && \
    conda install -n chemgrams_env pip && \
    /opt/conda/envs/chemgrams_env/bin/pip install https://github.com/kpu/kenlm/archive/master.zip --install-option="--max_order=10" && \
    /opt/conda/envs/chemgrams_env/bin/pip install deepsmiles && \
    /opt/conda/envs/chemgrams_env/bin/pip install networkx

WORKDIR /root
COPY resources/ chemgrams/resources/
COPY chemgrams/ chemgrams/chemgrams/
COPY examples/ chemgrams/examples/
COPY setup.py chemgrams/

WORKDIR /root/chemgrams
RUN /opt/conda/envs/chemgrams_env/bin/python setup.py install

ENV PATH="/root/kenlm/kenlm-master/build/bin:${PATH}"

WORKDIR /root/chemgrams/examples
CMD [ "/opt/conda/envs/chemgrams_env/bin/python", "lm_mcts_sequence_jscore_penalty_demo.py" ]
