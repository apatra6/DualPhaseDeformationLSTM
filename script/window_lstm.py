
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split

def read_data(dataset):
    OUT_FILE_PATH = "/home/sarthak/projects/seacas-exodus/lib/soudip_dataset/%s/out.csv" % dataset
    DATA_FILE_PATH = "/home/sarthak/projects/model_training/new_data/%s.csv" % dataset
    print(OUT_FILE_PATH)
    print(DATA_FILE_PATH)
    # max_vals == pd.read_csv('/max_vals.csv')
    # min_vals == pd.read_csv('/max_vals.csv')
    out_file = pd.read_csv(OUT_FILE_PATH)
    lim = 0.00
    timesteps = []
    time = []
    for i in range(len(out_file)):
        if(out_file["eff_strain"][i] >= lim):
            timesteps.append(i+1)
            time.append(out_file["time"][i])
            lim = lim + 0.01

    print("Reading %s \n" % dataset)
    # Reading data
    df = pd.read_csv(DATA_FILE_PATH)
    df_dropped = df.drop(columns=['strain_yy', 'phases', 'pressure', 'sdv22', 'sdv23',
                                  'total_strain_xy', 'elem_id', 'blk_id', 'total_stress_xx', 'total_stress_yy',
                                  'total_strain_xx', 'total_strain_yy' ])
    df_norm = (df_dropped-df_dropped.min())/(df_dropped.max()-df_dropped.min())
    df_norm.tail()
    scale_steps = int(np.ceil(len(out_file)/(len(df_dropped)/160000)))
    steps = np.ceil((np.array(timesteps)/scale_steps))
    df_steps = df_dropped[df_dropped["time"].isin(steps)] 
    df_steps_norm = (df_steps-df_steps.min())/(df_steps.max()-df_steps.min())
    df_steps_norm = df_steps_norm.drop(columns=['time'])
    df_group = df_steps_norm.groupby(["elem_x", "elem_y"])
    return df_group

def series_to_supervised(sequences, n_steps_in, n_steps_out, dropnan=True):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix: out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def structure_data(df_group):
    x_raw_list = []
    y_raw_list = []
    for name, group in df_group:
        strain_vals = group['eff_strain'].values
        stress_vals = group['vonmises'].values
        tri_vals = group['triaxiality'].values
        all_vals = np.stack((strain_vals, stress_vals, tri_vals), axis = 1 )
        x, y = series_to_supervised(all_vals, 2, 13)
        x_raw_list.append(x)
        y_raw_list.append(y)
    x_vals = np.concatenate(x_raw_list)
    y_vals = np.concatenate(y_raw_list)
    return x_vals, y_vals, x_raw_list, y_raw_list

df_group = read_data('AR1')
x_vals, y_vals, x_raw_list, y_raw_list = structure_data(df_group)

df_group_2 = read_data('test61')
x_vals_2, y_vals_2, x_raw_list_2, y_raw_list_2 = structure_data(df_group_2)

df_group_3 = read_data('test21')
x_vals_3, y_vals_3, x_raw_list_3, y_raw_list_3 = structure_data(df_group_3)

df_group_4 = read_data('test11')
x_vals_4, y_vals_4, x_raw_list_4, y_raw_list_4 = structure_data(df_group_4)

df_group_5 = read_data('test51')
x_vals_5, y_vals_5, x_raw_list_5, y_raw_list_5 = structure_data(df_group_5)

x_vals = np.concatenate((x_vals, x_vals_2, x_vals_3, x_vals_4, x_vals_5))
y_vals = np.concatenate((y_vals, y_vals_2, y_vals_3, y_vals_4, y_vals_5))
#n_steps_in, n_steps_out = 1, 1
print(x_vals.shape)
print(y_vals.shape)
n = x_vals.shape[0]

train_x = x_vals[0:int(0.7*n), :, :]
train_y = y_vals[0:int(0.7*n), :, :]
test_x = x_vals[int(0.7*n):int(0.9*n), :, :]
test_y = y_vals[int(0.7*n):int(0.9*n), :, :]
val_x = x_vals[int(0.9*n):, :, :]
val_y = y_vals[int(0.9*n):, :, :]
print(train_x.shape[2])

def get_compiled_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=(2, 3)))
    model.add(tf.keras.layers.RepeatVector(13))
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(train_x.shape[2])))
    model.compile(optimizer='adam', loss='mse')
    return model
model = get_compiled_model()
history = model.fit(train_x, train_y, epochs=20, shuffle=True)

print(model.evaluate(val_x, val_y))
base_path = '.'
model.save(base_path+'/the_model_5_datasets')
with open(base_path + '/model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))