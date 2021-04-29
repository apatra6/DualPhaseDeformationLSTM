import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split
class bcolors:
	OKBLUE = '\033[95m'
	OKGREEN= '\033[92m'
	ENDC = '\033[0m'


dataset_names = ['AR1', 'AR2', 'AR3', 'AR4', 'AR5',
				'test11', 'test12', 'test13', 'test14', 'test15', 
				'test21', 'test22', 'test23', 'test24', 'test25', 
				'test51', 'test52', 'test53', 'test54', 'test55',
				'test61', 'test62', 'test63', 'test64', 'test65',
				'test71', 'test72', 'test73', 'test74', 'test75']
MODEL_PATH = '/home/sarthak/projects/model_training/script'
model = tf.keras.models.load_model(MODEL_PATH+'/the_model')

for dataset in dataset_names:

	OUT_FILE_PATH = "/home/sarthak/projects/seacas-exodus/lib/soudip_dataset/%s/out.csv" % dataset
	DATA_FILE_PATH = "/home/sarthak/projects/model_training/new_data/%s.csv" % dataset
	print(OUT_FILE_PATH)
	print(DATA_FILE_PATH)

	out_file = pd.read_csv(OUT_FILE_PATH)
	lim = 0.01
	timesteps = []
	time = []
	for i in range(len(out_file)):
	    if(out_file["eff_strain"][i] > lim):
	        timesteps.append(i+1)
	        time.append(out_file["time"][i])
	        lim = lim + 0.01
	
	print("Reading %s \n" % dataset)
	# Reading data
	df = pd.read_csv(DATA_FILE_PATH)
	df_dropped = df.drop(columns=['pressure', 'sdv22', 'sdv23', 'total_strain_xy', 'elem_id', 'blk_id', 'total_stress_xx', 'total_stress_yy','total_strain_xx', 'total_strain_yy' ])
	df_norm = (df_dropped-df_dropped.min())/(df_dropped.max()-df_dropped.min())
	df_norm.tail()
	scale_steps = int(np.ceil(len(out_file)/(len(df_dropped)/160000)))
	steps = np.ceil((np.array(timesteps)/scale_steps))
	df_steps = df_dropped[df_dropped["time"].isin(steps)] 
	df_steps_norm = (df_steps-df_steps.min())/(df_steps.max()-df_steps.min())
	df_group = df_steps_norm.groupby(["elem_x", "elem_y"])

	print("Restructuring %s \n" % dataset)
	raw_list = []
	for name, group in df_group:
	    strain_vals = group['eff_strain'].values
	    stress_vals = group['vonmises'].values
	    tri_vals = group['triaxiality'].values
	    all_vals = np.stack((strain_vals, stress_vals, tri_vals), axis = 1 )
	    raw_list.append(all_vals)
	raw_vals = np.dstack(raw_list)
	raw_vals = np.rollaxis(raw_vals, -1)
	print(raw_vals.shape)

	x_vals = raw_vals[:, 0:len(steps)-1, :]
	y_vals = raw_vals[:, 1:len(steps), :]
	print("Data shape is ", x_vals.shape, y_vals.shape)
	all_pred_vals = model.predict(x_vals)

	PLOTS_PATH = "/home/sarthak/projects/model_training/script/predictions/%s" % dataset
	if not os.path.isdir(PLOTS_PATH):
		os.mkdir(PLOTS_PATH)
		print("Created %s" % PLOTS_PATH)

	variables = ["eff_strain", "vonmises", "triaxiality"]
	for variable in variables:
		var_name = variable
		plot_y_fe_scaled = y_vals[0:160000, -1, variables.index(variable)]
		plot_y_fe_scaled = (plot_y_fe_scaled*(df_steps[var_name].max() - df_steps[var_name].min())) + df_steps[var_name].min()
		# yy_levels = [0.00, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54]

		# Contour of FE Values
		xlist = np.arange(0.25, 100.25, 0.25)
		ylist = np.arange(0.25, 100.25, 0.25)
		X,Y = np.meshgrid(xlist, ylist)
		Z = plot_y_fe_scaled.reshape(400, 400)
		fig,ax=plt.subplots(1,1)
		plt.rcParams.update({'font.size': 14})
		mycmap1 = plt.get_cmap('rainbow')
		cp = ax.contourf(X, Y, Z, cmap=mycmap1)
		cbar = fig.colorbar(cp) # Add a colorbar to a plot
		ax.set_aspect('equal', adjustable='box')
		ax.set_xlabel('x ($\mu m$)')
		ax.set_ylabel('y ($\mu m$)')
		plt.savefig(PLOTS_PATH+'/contour_cpfe_'+var_name+'.eps', format='eps')

		plot_pred_scaled = all_pred_vals[0:160000, -1, variables.index(variable)]
		plot_pred_scaled = (plot_pred_scaled*(df_steps[var_name].max() - df_steps[var_name].min())) + df_steps[var_name].min()
		# yy_levels = [0.00, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54]

		# Contour of LSTM Values
		xlist = np.arange(0.25, 100.25, 0.25)
		ylist = np.arange(0.25, 100.25, 0.25)
		X,Y = np.meshgrid(xlist, ylist)
		Z = plot_y_fe_scaled.reshape(400, 400)
		fig,ax=plt.subplots(1,1)
		plt.rcParams.update({'font.size': 14})
		mycmap1 = plt.get_cmap('rainbow')
		cp = ax.contourf(X, Y, Z, cmap=mycmap1)
		cbar = fig.colorbar(cp) # Add a colorbar to a plot
		ax.set_aspect('equal', adjustable='box')
		ax.set_xlabel('x ($\mu m$)')
		ax.set_ylabel('y ($\mu m$)')
		plt.savefig(PLOTS_PATH+'/contour_cpfe_'+var_name+'.eps', format='eps')

		# Contour of Error Values
		plot_diff = (plot_pred_scaled - plot_y_fe_scaled)
		xlist = np.arange(0.25, 100.25, 0.25)
		ylist = np.arange(0.25, 100.25, 0.25)
		X,Y = np.meshgrid(xlist, ylist)
		Z = plot_diff.reshape(400, 400)
		fig,ax=plt.subplots(1,1)
		plt.rcParams.update({'font.size': 14})
		mycmap1 = plt.get_cmap('rainbow')
		cp = ax.contourf(X, Y, Z, cmap=mycmap1)
		fig.colorbar(cp) 
		ax.set_aspect('equal', adjustable='box')
		ax.set_xlabel('x ($\mu m$)')
		ax.set_ylabel('y ($\mu m$)')
		plt.savefig(PLOTS_PATH+'/contour_error_'+var_name+'.eps', format='eps')

		r2 = []
		test_vec = plot_pred_scaled.reshape(1, 160000)[0]
		pred_vec = plot_y_fe_scaled.reshape(1, 160000)[0]
		r2.append(r2_score(pred_vec, test_vec))
		err1 = sum(abs(pred_vec-test_vec))/160000
		print(r2, err1)

		plt.figure()
		plt.rcParams.update({'font.size': 14})
		a = plt.axes(aspect='equal')
		a.text(min(plot_pred_scaled), 0.9*max(plot_pred_scaled), "$R^2$ = %0.2f"%(r2[0]), fontsize=15)
		plt.scatter(plot_y_fe_scaled.reshape(1, 160000), plot_pred_scaled.reshape(1, 160000))
		xp = [min(plot_pred_scaled),max(plot_pred_scaled)]
		yp = xp
		plt.plot(xp,yp, 'r')
		plt.xlabel('FE Values')
		plt.ylabel('LSTM Predictions')
		plt.savefig(PLOTS_PATH+"/"+var_name+".eps", format='eps')
	plt.close('all')
	print(bcolors.OKGREEN + "Done with dataset % s" % (dataset) + bcolors.ENDC)
