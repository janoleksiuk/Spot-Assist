import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pandas as pd
import torch
from preprocesiing import load_data_train
from sklearn import preprocessing

# input fun returns code:
# 0 - error occured during reading csv
# 1 - successful data acquirenment from csv

def input(trainpath, isTrain = True):
	d = {'sitting': 0, 'standing': 1, 'sitting_1hand': 2, 'standing_1hand': 3}

	osize=57

	try:
		file_out_t = pd.read_csv(trainpath)
	except pd.errors.EmptyDataError:
		return pd.DataFrame(), 0

	sizetrain = file_out_t.iloc[0:, 0:osize].values.shape[0]
	x_train = file_out_t.iloc[0:sizetrain, 0:osize].values
	y_train = file_out_t.iloc[0:sizetrain, osize].values
	for n in range(len(y_train)):
		if y_train[n]=="sittting" :
			y_train[n]="sitting"
	label=y_train
	t=[]
	for k in y_train:
		t.append(d[str(k)])
	y_train=np.array(t)
	mm=[]
	for i in range(len(label)):
		mm.append(str(label[i])+"-"+str(y_train[i]))
	table1=np.unique(label)
	table2=np.unique(y_train)
	
	if isTrain:
		data = {'x_train': x_train,  
				'y_train': y_train}
	
	else:
		data = {'x_test': x_train,  
				'y_test': y_train}

	return data, 1
