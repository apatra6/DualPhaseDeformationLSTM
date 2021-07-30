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