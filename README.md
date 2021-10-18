# A Machine Learning-Based Surrogate Modeling Framework for Predicting the History-Dependent Deformation of Dual Phase Microstructures
## Sarthak Khandelwal, Soudip Basu, Anirban Patra
## Department of Metallurgical Engineering and Materials Science, Indian Institute of Technology Bombay, Mumbai, India

## Abstract
A Machine Learning (ML)-based surrogate modeling framework is developed to predict the heterogeneous deformation behavior of dual phase microstructures. The deformation is first simulated using a dislocation density-based J2 plasticity Finite Element (FE) model, whose results form the basis for surrogate model training and validation. Long Short Term Memory (LSTM)-based ML models, with different architectures, are employed to predict the spatio-temporal evolution of three output variables: effective strain, von Mises effective stress, and the stress triaxiality ratio. Two metrics, the mean average error (MAE) and the coefficient of determination, $R^2$, are used to assess the performance of the models and different architectures. Based on our analysis, the LSTM model is generally found to predict the spatio-temporal deformation fields with reasonable accuracy, even for untrained microstructures with varying microstructural attributes and random instantiations. The LSTM model is also used to predict aggregate properties, such as the stress-strain response and the strain partitioning in the dual phase microstructures.

## Training
[LSTM-based RNN](https://github.com/TheFlash98/model_training/blob/master/window-lstm.ipynb): A LSTM-based RNN model capable of predicting the microstructure evolution during deformation of dual phase microstructures by just looking at the microstructure after 1% strain. We have predicted three variables using this model: effective strain, effective stress and stress triaxiality ratio. 

The model analysis, predictions and results for different microstructures can be produced using [this](https://github.com/TheFlash98/model_training/blob/master/window-lstm-plot-analysis.ipynb).

Here's a brief introduction to what each file in this repository does

- `script/pred_all_dataset.py`: takes a particular model and makes predictions for all the given microstructures using the model. It stores the results in a separate folder for each microstructure with appropriate names.
- `script/pred_all_dataset.py`: this script consists of the LSTM model's architecture and training. 5 mircostructure's data has been used to train the model. The script saves the model and plots the evolution of loss with each epoch.
- `window-lstm-predictions.ipynb`: this notebook consists of the LSTM model's architecture and training with proper explaination for each code block
- `window-lstm-training.ipynb`: this notebook consists of the plots made for the analysis of the predictions made using this model.
