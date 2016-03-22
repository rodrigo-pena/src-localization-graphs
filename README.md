# src-localization-graphs
Implementation of algorithms for source localization on graphs via l1 regularization and spectral graph theory.

## Installation
First, make sure to download and install the [GSPBox](https://lts2.epfl.ch/gsp/), and [MatlabBGL](http://dgleich.github.io/matlab-bgl/). Then, in MATLAB, just change the current directory to the src-localization-graphs folder and run src\_loc\_start.m to add the needed dependencies to path.

If you want to work with John Snow's GIS data, and the ETEX data (see paper in the References section below), you will need to download and install the [load-data](https://github.com/rodrigo-pena/load-data) toolbox.

##Folders
###root
Contains the main functions of the toolbox. Use function **alt\_opt.m** to simultaneously learn the source locations and the diffusion kernel. If the source locations are known, use **learn\_param\_kernel.m** to learn the diffusion kernel. If the diffusion kernel is known, use **learn\_sparse\_signal.m** to learn the source locations.

###3rd_party/
Contains a third party function to plot error bars over a surface embedded in 3-dimensional space.

###demos/
Contains demonstrations of what can be done with the toolbox. The names should be self explanatory.

###experiments/
Contains the scripts for reproducing the experiments in [1].

###solvers/
Contains the numerical solvers used in the alternate optimization problem: FISTA, and Newton's method.

###utils/
Contains useful functions for the toolbox, such as error measures, parameter estimation and initialization strategies, etc.
##References
[1] R. Pena, X. Bresson, and P. Vandergheynst. "Source Localization on Graphs via l1 Regularization and Spectral Graph Theory." Preprint, 2016.