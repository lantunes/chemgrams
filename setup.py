from setuptools import setup

setup(name="chemgrams",
      version="0.1.1",
      description="Chemgrams, A library for working with n-gram chemical language models, for Python.",
      license="Apache License 2.0",
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
      ],
      url='http://github.com/lantunes/chemgrams',
      author="Luis M. Antunes",
      author_email="lantunes@gmail.com",
      packages=["chemgrams", "chemgrams.jscorer", "chemgrams.sascorer", "chemgrams.qedscorer", "chemgrams.logger",
                "chemgrams.tanimotoscorer", "chemgrams.cyclescorer", "chemgrams.training", "chemgrams.queryscorer"],
      package_data={'chemgrams.sascorer': ['fpscores.pkl.gz'], 'chemgrams.training' : ['train_kenlm.sh']},
      include_package_data=True,
      python_requires='>3.5.2',
      install_requires=["nltk == 3.4.5", "deepsmiles", "networkx"])
