# src-localization-graphs
MATLAB implementation of an algorithm for source localization on graphs via l1 regularization and spectral graph theory. As presented in our paper:

R. Pena, X. Bresson, and P. Vandergheynst. ["Source Localization on Graphs via l1 Regularization and Spectral Graph Theory"][arxiv]. 12th IEEE Image, Video, and Multidimensional Signal Processing Workshop (IVMSP), 2016.

The code is released under the terms of the [MIT license](LICENSE.txt). Please cite the paper above if you use it.

[![DOI](https://zenodo.org/badge/54215534.svg)](https://zenodo.org/badge/latestdoi/54215534)

## Requirements
Make sure to download and install the following toolboxes:
* [GSPBox](https://lts2.epfl.ch/gsp/)
* [MatlabBGL](http://dgleich.github.io/matlab-bgl/)

## Installation
1. Clone this repository.

  ```sh
   git clone https://github.com/rodrigo-pena/src-localization-graphs
   cd src-localization-graphs
   ```
   
2. Install the dependencies mentioned on the previous section.

3. In MATLAB, change the current directory to the *src-localization-graphs* folder and run [src_loc_start.m](https://github.com/rodrigo-pena/src-localization-graphs/blob/master/src_loc_start.m) to automatically add all the subfolders to your path.

4. Run [demo_alternate_optimization.m](https://github.com/rodrigo-pena/src-localization-graphs/blob/master/demos/demo_alternate_optimization.m) to see if everything works fine.

## Remark: John Snow's GIS, and ETEX data
If you want to work with John Snow's GIS, or the ETEX data (as used in the [paper][arxiv]), you will need to clone another repository, [load-data](https://github.com/rodrigo-pena/load-data), and follow the installation instructions in the README therein.

## Folders
### [root](https://github.com/rodrigo-pena/src-localization-graphs)
Contains the main functions of the toolbox. Use function [alt_opt.m](https://github.com/rodrigo-pena/src-localization-graphs/blob/master/alt_opt.m) to simultaneously learn the source locations and the diffusion kernel. If the source locations are known, use [learn_param_kernel.m](https://github.com/rodrigo-pena/src-localization-graphs/blob/master/learn_param_kernel.m) to learn the diffusion kernel. If the diffusion kernel is known, use [learn_sparse_signal.m](https://github.com/rodrigo-pena/src-localization-graphs/blob/master/learn_sparse_signal.m) to learn the source locations.

### [3rd_party/](https://github.com/rodrigo-pena/src-localization-graphs/tree/master/3rd_party)
Contains a third party function to plot error bars over a surface embedded in 3-dimensional space. Used only in [testbench.m](https://github.com/rodrigo-pena/src-localization-graphs/blob/master/experiments/testbench.m).

### [demos/](https://github.com/rodrigo-pena/src-localization-graphs/tree/master/demos)
Contains demonstrations of what can be done with the toolbox. The names should be self-explanatory.

### [experiments/](https://github.com/rodrigo-pena/src-localization-graphs/tree/master/experiments)
Contains the scripts for reproducing the experiments in [paper][arxiv].

### [solvers/](https://github.com/rodrigo-pena/src-localization-graphs/tree/master/solvers)
Contains the numerical solvers used in the alternate optimization problem: FISTA, and Newton's method.

### [utils/](https://github.com/rodrigo-pena/src-localization-graphs/tree/master/utils)
Contains useful functions for the toolbox, such as error measures, parameter estimation and initialization strategies, etc.

## Usage
To use the source localization algorithms on your data, you'll need:

1. A graph stored in [GSPBox](https://lts2.epfl.ch/gsp/)-compatible format. 
2. A source indicator vector, with non-zero elements on the indices representing the source nodes, and zeros everywhere else.
3. An observation vectorn whose entries encode the signal value at each node.
4. A parametric kernel representing the diffusion process.

See any of [demos](https://github.com/rodrigo-pena/src-localization-graphs/tree/master/demos) for examples on how to call the methods. Please get in touch if you are unsure about how to adapt the code to your settings.

## References
R. Pena, X. Bresson, and P. Vandergheynst. ["Source Localization on Graphs via l1 Regularization and Spectral Graph Theory"][arxiv]. 12th IEEE Image, Video, and Multidimensional Signal Processing Workshop (IVMSP), 2016.

[arxiv]: https://arxiv.org/abs/1603.07584
