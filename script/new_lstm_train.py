import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split

out_file = pd.read_csv('./AR1/out.csv')
lim = 0.01
timesteps = []
time = []
for i in range(len(out_file)):
    if(out_file["eff_strain"][i] > lim):
        timesteps.append(i+1)
        time.append(out_file["time"][i])
        lim = lim + 0.01
steps = np.ceil((np.array(timesteps)/50))

out_file_ar2 = pd.read_csv('./test21/out.csv')
lim_ar2 = 0.01
timesteps_ar2 = []
time_ar2 = []
for i in range(len(out_file_ar2)):
    if(out_file_ar2["eff_strain"][i] > lim_ar2):
        timesteps_ar2.append(i+1)
        time_ar2.append(out_file_ar2["time"][i])
        lim_ar2 = lim_ar2 + 0.01
steps_ar2 = np.ceil((np.array(timesteps_ar2)/50))

out_file_ar3 = pd.read_csv('./test61/out.csv')
lim_ar3 = 0.01
timesteps_ar3 = []
time_ar3 = []
for i in range(len(out_file_ar3)):
    if(out_file_ar3["eff_strain"][i] > lim_ar3):
        timesteps_ar3.append(i+1)
        time_ar3.append(out_file_ar3["time"][i])
        lim_ar3 = lim_ar3 + 0.01
steps_ar3 = np.ceil((np.array(timesteps_ar3)/50))

print("Dataset 1 has % i strain steps" % len(steps))
print("Dataset 2 has % i strain steps" % len(steps_ar2))
print("Dataset 3 has % i strain steps" % len(steps_ar3))
steps_all = min(len(steps), len(steps_ar2))
steps_ar3 = steps_ar3[0:steps_all]
steps_ar2 = steps_ar2[0:steps_all]
steps = steps[0:steps_all]
print("Going forward with % i strain steps" % steps_all)
print("")

print("Reading all the data")
print("")
# Reading data
df = pd.read_csv('./AR1/AR1.csv')
df_dropped = df.drop(columns=['pressure', 'sdv22', 'sdv23', 'total_strain_xy', 'elem_id', 'blk_id', 'total_stress_xx', 'total_stress_yy',
	'total_strain_xx', 'total_strain_yy' ])
df_norm = (df_dropped-df_dropped.min())/(df_dropped.max()-df_dropped.min())
df_norm.tail()
df_steps = df_dropped[df_dropped["time"].isin(steps)] 

df_ar2 = pd.read_csv('./test21/test21.csv')
df_dropped_ar2 = df_ar2.drop(columns=['pressure', 'sdv22', 'sdv23', 'total_strain_xy', 'elem_id', 'blk_id', 'total_stress_xx', 
	'total_stress_yy','total_strain_xx', 'total_strain_yy' ])
df_ar2_norm = (df_ar2-df_ar2.min())/(df_ar2.max()-df_ar2.min())
df_ar2_norm.tail()
df_steps_ar2 = df_dropped_ar2[df_dropped_ar2["time"].isin(steps_ar2)] 

df_ar3 = pd.read_csv('./test61/test61.csv')
df_dropped_ar3 = df_ar3.drop(columns=['pressure', 'sdv22', 'sdv23', 'total_strain_xy', 'elem_id', 'blk_id', 'total_stress_xx', 
	'total_stress_yy','total_strain_xx', 'total_strain_yy' ])
df_ar3_norm = (df_ar3-df_ar3.min())/(df_ar3.max()-df_ar3.min())
df_ar3_norm.tail()
df_steps_ar3 = df_dropped_ar3[df_dropped_ar3["time"].isin(steps_ar3)]

max_vals = []
min_vals = []
max_min_index = ['eff_strain', 'strain_yy', 'time', 'triaxiality', 'vonmises', 'elem_x', 'elem_y', 'phases']
for i in max_min_index:
    max_vals.append(max(df_steps.max()[i], df_steps_ar2.max()[i], df_steps_ar3.max()[i]))
    min_vals.append(min(df_steps.min()[i], df_steps_ar2.min()[i], df_steps_ar3.min()[i]))

max_series = pd.Series(max_vals)
min_series = pd.Series(min_vals)
max_series.index = max_min_index
min_series.index = max_min_index
max_series.to_csv('./max_vals.csv')
min_series.to_csv('./min_vals.csv')

print("Normalising data")
print("")
df_steps_norm = (df_steps - min_series)/(max_series - min_series)
df_group = df_steps_norm.groupby(["elem_x", "elem_y"])

df_steps_norm_ar2 = (df_steps_ar2 - min_series)/(max_series - min_series)
df_group_ar2 = df_steps_norm_ar2.groupby(["elem_x", "elem_y"])

df_steps_norm_ar3 = (df_steps_ar3 - min_series)/(max_series - min_series)
df_group_ar3 = df_steps_norm_ar3.groupby(["elem_x", "elem_y"])

print("Restructuring data")
raw_list = []
for name, group in df_group:
    strain_vals = group['eff_strain'].values
    stress_vals = group['vonmises'].values
    tri_vals = group['triaxiality'].values
    all_vals = np.stack((strain_vals, stress_vals, tri_vals), axis = 1 )
    raw_list.append(all_vals)

for name, group in df_group_ar2:
    strain_vals = group['eff_strain'].values
    stress_vals = group['vonmises'].values
    tri_vals = group['triaxiality'].values
    all_vals = np.stack((strain_vals, stress_vals, tri_vals), axis = 1 )
    raw_list.append(all_vals)

for name, group in df_group_ar3:
    strain_vals = group['eff_strain'].values
    stress_vals = group['vonmises'].values
    tri_vals = group['triaxiality'].values
    all_vals = np.stack((strain_vals, stress_vals, tri_vals), axis = 1 )
    raw_list.append(all_vals)

raw_vals = np.dstack(raw_list)
raw_vals = np.rollaxis(raw_vals, -1)

x_vals = raw_vals[:, 0:steps_all-1, :]
y_vals = raw_vals[:, 1:steps_all, :]
print("Shape of the data is " , x_vals.shape, y_vals.shape)	

train_x_vals, test_x_vals, train_y_vals, test_y_vals = train_test_split(x_vals, y_vals, test_size=0.30)
print(train_x_vals.shape, test_x_vals.shape)

neurons  = 50

def get_compiled_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(neurons, activation='tanh', input_shape=(steps_all - 1, 3)))
    model.add(tf.keras.layers.RepeatVector(steps_all - 1))
    model.add(tf.keras.layers.LSTM(neurons, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(neurons, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(neurons, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(neurons, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(neurons, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(neurons, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(neurons, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3)))
    model.compile(optimizer='adam', loss='mse')
    return model

model = get_compiled_model()
history = model.fit(train_x_vals, train_y_vals, epochs=50, shuffle=False)

plt.figure()
plt.xlabel("No. of Epochs")
x = [1, 2, 3, 4, 5]
plt.ylabel("MSE")
plt.plot(history.history['loss'], label="MSE (training data)")
plt.savefig(base_path+"/lstm-training_process.eps", format="eps")

model.save('./the_model')
with open(base_path + '/model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
