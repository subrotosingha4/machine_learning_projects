# Mindwandering Deep Learning Classifier

This repository contains reproducible research results containing
code, data and manuscript preparation for a replication and extension
of a mindwandering classifier.  The original data and machine learning
models being used and replicated in this paper are from:

- Bixler, R., & D’Mello, S. (2015). Automatic gaze-based detection of
  mind wandering with metacognitive awareness. In International
  conference on user modeling, adaptation, and personalization
  (Vol. 9146, pp. 31–43). Springer, Cham.
- Faber, M., Bixler, R., & D’Mello, S. K. (2018). An automated
  behavioral measure of mind wandering during computerized
  reading. Behavior Research Methods, 50(1), 134-150.

In particular this current work replicates the standard machine
learning classifier results from Faber, Bixler & D'Mello (2018),
then extends the results to deep learning classifiers on the
same mind wandering dataset.


# Getting Started

The files in this project are self contained to reproduce the results, figures
and tables for the target paper.  Currently components are built and tested
individually.

This repository currently assumes an environment with the following tools
available:

TODO: Make this more defniitive.  Also add in a project setup script and/or
find best practices to specify these so can perform needed installs automatically
to reproduce.

- Python version 3.x, used 
  [Anaconda Distribution for Python 3.8](https://www.anaconda.com/products/individual) 
  in repository development.
  - From standard anaconda distribution, expect standard scientific python
    library stack, `matplotlib`, `numpy`, `scipy`
  - Need also `keras`, `tensorflow-gpu`,  `scikit-learn` and
    `imbalanced-learn` which may need to be added separatly
    ```
	$ conda install keras tensorflow-gpu scikit-learn 
	$ conda install -c conda-forge imbalanced-learn
	```
- Gnu make build and development tools
  ```
  # from standard apt based linux package manager
  $ sudo apt install build-essential
  ```
- \LaTex tools, this repository used `texlive-latex-base`, `texlive-latex-recommended`
  `texlive-latex-extra` and `biber` \LaTeX packages.
  ```
  # in an apt based linux package manager system
  $ sudo apt install texlive-latex-base texlive-latex-recommended texlive-latex-extra biber
  ```

# Reproducable Research

This repository follows open science and reproducible research best practices.
The repository is structured to rebuild and reproduce all results, from data
cleaning and analysis, through model training and final generation of figures
and manuscript, from a single build system.  In general, from the top level
of the repository you can do

```
$ make clean
$ make
```

To regenerate all results, figures, tables and final manuscript from scratch.
Each subproject directory can also be (re)built separately using make.
The research products are described next and built in the following order.

### tests

Reusable python modules and source code are found in the `src` directory, as well
as scripts used in document production and results production.

The `mindwandering` library contains reusable code used to clean data, set up
training pipelines, and normalize and format data for reproducing results.

To test data cleaning, pipeline and data normalization routines, run the unit tests
for the `mindwandering` module:

```
$ cd src
src$ make
```

### models

All results of model training are done in the models subdirectory.  This 
subproject uses scikit-learn and keras to build and train various
estimators.  Results of all models trained are saved as pickle .pkl
files, for use in data analysis and visualization.  The models can
take a long time to build.  Keras models should make use of gpu
resources for training when available.  To rebuild all models
do a build in the models subdirectory:

```
$ cd models
models$ make
```

### figures

All figures and data visualizations are generated in the figures subdirectory.
This code mainly uses matplotlib and seaborn for generating visualizaitons
of the trained estimator results.  To rebuild all figures do a build
in the figures subdirectory:

```
$ cd figures
figures$ make
```

### paper

The `paper` subdirectory uses pweave and pandoc to generate a publication
ready document from (pandoc) markdown source.  It converts pandoc markdown
to \LaTeX markup, then \LaTeX markup to final pdf document.  

The markdown file may incorporate \LaTeX markup for math notation, tables
and bibliography and reference generation.  Pandoc code snippts can be
inserted in documents, and output directly generated as part of making the
paper.

The paper does use figures and tables deposited in those subdirectories by
other build processes.  If figures or tables are updated, regenerate the
paper to pick them up.

The basic workflow to generate the pdf paper from source is:

```
$ cd paper
paper$ make
```

