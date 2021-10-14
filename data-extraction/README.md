## Data Extraction

The simulation results of the FE model are output to an EXODUS data file. The EXODUS
format is a binary file which is used for FE pre- and post-processing. Data used to define
the finite element mesh, along with both the boundary conditions and load application
points are includes in the file. The EXODUS format is advantageous as it combines the
mesh data as well as the results data in a single file. This assures the user that the
results are consistent with the model. However, to access this data in a usable and
easily interpretable way, special software are used. For the purpose of this project, we
have used the Sandia Engineering Analysis Code Access System (SEACAS) developed by
Sjaardema. The library consists of packages which can convert an EXODUS file’s data
to different formats like text file and MATLAB data files. The Python package has been
used for this project which makes reading the data from an EXODUS file viable through a
python script. The package provides a number of predefined functions to extract different
information from the EXODUS file. Using these functions, we have extracted all the data
from the EXODUS file to a csv format. The csv format has all the element variable values at every time-step for every element. The script is written in Python 2.7.17 and take approximately 10 minutes to execute. The time taken by the script varies with the mesh
size and number of variables in the exodus file. The script can be found [here](https://github.com/TheFlash98/model_training/blob/master/data-extraction/make_data.py).
