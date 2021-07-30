# Predicting the Time Dependent Deformation of Dual Phase

## Abstract

Surrogate machine learning models are proposed for modeling the elastic-plastic deforma-
tion of dual phase microstructures. The deformation is first simulated using a J2 plasticity
model, whose results form the basis for surrogate model development. A simple Artificial
Neural Network (ANN) model is applied to make the predictions for von-Mises effective
stress, stress triaxiality ratio and effective strain. We further test the performance for a
Long Short Term Memory (LSTM) based Recurrent Neural Network (RNN) and compare
the results obtained from the two models. For both the approaches, multiple models were
trained and the one, with least overall error, was used to make the predictions. There
is a significant increase in the accuracy with the latter model, which has also been used
to predict the deformation contours in untrained microstructures, as well the aggregate
stress-strain response and the strain partitioning between the two phases.

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

## Training

A different number of models were trained with different structures during the course of this project. The codes for each of them can be found in this repository.

- [Artificial Neural Networks](https://github.com/TheFlash98/model_training/blob/master/one-variable-ann.ipynb): An ANN model is trained to predict only the effective strain of a given microstrcuture at a particular timestep using the x,y coordinates, phase information and the total applied strain at that timestep. The model has 8 layers with 128, 64, 32, 16, 8, 4, 2, 1 layers respectively. The notebook also contains predictions made using the model.

- [LSTM-based RNN](https://github.com/TheFlash98/model_training/blob/master/window-lstm.ipynb): A LSTM-based RNN model capable of predicting the entire evolution of a microstructure by just looking at the microstrcture at 1% strain. Using this model we have predicted three variable, effective strain, vonmises stress and stress triaxiality. The model analysis, predictions and results over different microstructures can be produced using [this](https://github.com/TheFlash98/model_training/blob/master/window-lstm-plot-analysis.ipynb).