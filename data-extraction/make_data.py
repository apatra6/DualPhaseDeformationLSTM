from exodus import *
import numpy as np
from itertools import izip_longest
import pandas as pd
import csv
import time

# Path to the exodus file you want to extract data from
EXODUS_FILE = "soudip_dataset/AR2/out.e" 

exodusHandle = exodus(EXODUS_FILE, 'r', array_type='numpy')
timesteps = exodusHandle.num_times()
#timesteps = 1

def grouper(iterable, n, fillvalue=None):
	args = [iter(iterable)] * n
	return izip_longest(fillvalue=fillvalue, *args)

def find_elem_id(arr):
	'''
	Input information about the connectivity of four nodes, i.e., the elements they
	are a part of. The intersection of all those elements ids gives the unique
	element which is made up of the four nodes. 
	'''
	result = set(arr[0])
	for curr_set in arr[1:]:
		result.intersection_update(curr_set)
	return list(result)[0]

def find_elem_loc(nodes):
	'''
	Inputs 4 nodes which make up an elements and calculates the coordinates for those 
	elements. (x,y) coordinates are only defined for nodes, elements' coordinates are
	caculated by computing the centeroid of the 4 nodes.
	'''
	coord_map = map(exodusHandle.get_coord, nodes)
	loc = res = [sum(i[0]/4 for i in coord_map), sum(i[1]/4 for i in coord_map)]
	return loc

def get_stress_strain_values(grain):
	'''
	Take a grain and returns all element variable values along with one global variable:
	'total_strain_yy', as a pandas dataframe. 
	'''
 	elem_var_names = exodusHandle.get_element_variable_names()
	strain_yy = exodusHandle.get_global_variable_values('total_strain_yy')
	df2 = {"strain_yy":[], "time": []}
 	for time in range(0,timesteps):
 		strain_yy = exodusHandle.get_global_variable_value('total_strain_yy', time)
		for elem_name in elem_var_names:
			new_val = exodusHandle.get_element_variable_values(grain, elem_name,time)
			if elem_name in df2.keys():
				df2[elem_name] = np.concatenate((df2[elem_name], new_val))
			else:
				df2[elem_name]= new_val
			num_elem = len(new_val)
		df2["strain_yy"] = np.concatenate((df2["strain_yy"], [strain_yy]*num_elem))
		df2["time"] = df2["time"] + [time]*num_elem
	return pd.DataFrame.from_dict(df2)

def main():
	# Gets id of all the block / grain in the microstructure
	elem_block_ids = exodusHandle.get_elem_blk_ids() 
	elem_variable_names = exodusHandle.get_element_variable_names


	connectivity = [] # Nodes which constitue a given element
	localNodeToLocalElems = [] # Elem ids of which the node is a part
	localElemToLocalElems = [] # Lists the neighbours of a given element
	# generates a list of lists to go from local elem id to connected local elem ids
	collectLocalElemToLocalElems(exodusHandle, connectivity, localNodeToLocalElems, localElemToLocalElems)
	
	print(connectivity[0],localElemToLocalElems[0], localNodeToLocalElems[1])

	node_ids = exodusHandle.get_node_id_map()
	elem_ids = exodusHandle.get_elem_id_map()
	blk_ids = exodusHandle.get_elem_blk_ids()

	# Get all the necessary variables for each grain (block) and store them in a map.
	stress_strain_values_map = map(get_stress_strain_values, blk_ids)
	print "Done creating map"
	

	for blk_id in blk_ids:
		# Get the nodal connectivity, number of elements, and number of nodes per element for a single block
		elem_conn, num_blk_elems, num_elem_nodes = exodusHandle.get_elem_connectivity(blk_id)

		elem_var_df = get_stress_strain_values(blk_id)
		elem_id = []
		elem_x = [] # x coordinate for an element
		elem_y = [] # y coordinate for an element
		phases = [] # phase of each element
		for node_1, node_2, node_3, node_4 in grouper(elem_conn, 4):
			node_to_elems = [localNodeToLocalElems[node_1], localNodeToLocalElems[node_2]
			, localNodeToLocalElems[node_3], localNodeToLocalElems[node_4]]
			elem_id.append(find_elem_id(node_to_elems))
			temp = find_elem_loc(connectivity[elem_id[-1]])
			elem_x.append(temp[0])
			elem_y.append(temp[1])
			# The phase changes after a particular block id. Obtained from paraview. 
			if blk_id > 429:
				phases.append(1) 
			else:
				phases.append(0)
		# Adding rows of data for each element at each timestep.
		stress_strain_values_map[blk_id-1]['elem_id'] =  elem_id*timesteps
		stress_strain_values_map[blk_id-1]['elem_x'] = elem_x*timesteps
		stress_strain_values_map[blk_id-1]['elem_y'] = elem_y*timesteps
		stress_strain_values_map[blk_id-1]['phases'] = phases*timesteps
		stress_strain_values_map[blk_id-1]['blk_id'] = [blk_id]*timesteps*len(elem_id)

	# Print misc information to ensure script run properly
	print len(stress_strain_values_map[0]['phases']), len(stress_strain_values_map[0]['strain_yy'])
	print len(stress_strain_values_map[10]['phases']), len(stress_strain_values_map[10]['strain_yy'])
	print stress_strain_values_map[0].keys()
	for key in stress_strain_values_map[0].keys():
		print key, len(stress_strain_values_map[0][key])

	# Convert the map of pandas dataframes to a single dataframe and then convert it to csv
	all_data = pd.concat(stress_strain_values_map)
	print(all_data.dtypes)
	all_data.to_csv('~/projects/model_training/data/AR2/out_soudip.csv', index=False)
	


if __name__ == '__main__':
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))






