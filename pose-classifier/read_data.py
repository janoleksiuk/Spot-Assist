import numpy as np
import pandas as pd

# preparing input for adapted PNN
def input(input, isTrain = True, isCSV = False):
	d = {'sitting': 0, 'standing': 1, 'sitting_1hand': 2, 'standing_1hand': 3}
	osize = 57 # num of input variables

	if isCSV:
		try:
			file_out_t = pd.read_csv(input)
		except pd.errors.EmptyDataError:
			return pd.DataFrame(), 0
	else:
		file_out_t = pd.DataFrame(input)
		file_out_t["label"] = "standing"

	sizetrain = file_out_t.iloc[0:, 0:osize].values.shape[0]
	x_train = file_out_t.iloc[0:sizetrain, 0:osize].values
	y_train = file_out_t.iloc[0:sizetrain, osize].values
	
	# reference set may containg misspelled word 'sitting'
	for n in range(len(y_train)):
		if y_train[n] == "sittting" :
			y_train[n] = "sitting"
	
	label = y_train
	t = []
	for k in y_train:
		t.append(d[str(k)])
	y_train = np.array(t)
	mm = []

	for i in range(len(label)):
		mm.append(str(label[i]) + "-" + str(y_train[i]))

	# splitting depending on whether it is reference or prod data
	if isTrain:
		data = {'x_train': x_train,  
				'y_train': y_train}
	else:
		data = {'x_test': x_train,  
				'y_test': y_train}

	return data, 1