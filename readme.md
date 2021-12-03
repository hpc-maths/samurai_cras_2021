This repository contains all the material needed to reproduce the numerical simulations described in the article ["Does the multiresolution lattice Boltzmann method allow to deal with waves passing through mesh jumps?"](https://hal.archives-ouvertes.fr/hal-03235133v1).

These simulations are perform using [samurai](https://github.com/hpc-maths/samurai): an open source software written in C++ which provides a new data structure based on intervals and algebra of sets to handle efficiently adaptive mesh refinement methods based on a cartesian grid.

You can reproduce the figures by clicking on the binder logo just below which will opens a jupyter notebook.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hpc-maths/samurai_cras_2021/HEAD?labpath=plot_results.ipynb)

If you want run the experiments locally, the process is the following

- Install conda or mamba (https://github.com/conda-forge/miniforge)
- Clone this repo and go into
- Install the environment using the command line

```bash
conda env create -f binder/environment.yml
```

or

```bash
mamba env create -f environment.yml
```

- Activate the environment

```bash
conda activate samurai-cras-2021
```

- Install samurai and build the project

```bash
bash binder/postBuild
```

- Open the jupyter notebook

```bash
jupyter lab plot_results.ipynb
```

- And execute all the cells