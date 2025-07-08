import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pandas as pd
import torch
from preprocesiing import load_data_train
from sklearn import preprocessing
def input(trainpath,testpath):
	d = {'sitting': 0, 'standing': 1, 'sitting_1hand': 2, 'standing_1hand': 3}

	# train_data = "/mnt/storage/buildwin/soat/label-new_warsadata/"
	# test_data = "/mnt/storage/buildwin/soat/test-warsaw/"
	# x_train, y_train, trl2 = load_data_train(train_data, 1)
	# x_test, y_test, tel2 = load_data_train(test_data,  0)

	osize=57
	file_out_t = pd.read_csv(trainpath)
	file_out = pd.read_csv(testpath)
	sizetrain = file_out_t.iloc[0:, 0:osize].values.shape[0]
	sizetest=file_out.iloc[0:, osize].values.shape[0]
	x_train = file_out_t.iloc[0:sizetrain, 0:osize].values
	print( sizetrain,sizetest)
	# x_train=torch.tensor(x_train, dtype=torch.float32).numpy()
	y_train = file_out_t.iloc[0:sizetrain, osize].values
	for n in range(len(y_train)):
		if y_train[n]=="sittting" :
			y_train[n]="sitting"
	label=y_train
	t=[]
	for k in y_train:
		t.append(d[str(k)])
	y_train=np.array(t)
	# le = preprocessing.LabelEncoder()
	# y_train=torch.as_tensor(le.fit_transform(y_train)).numpy()
	mm=[]
	for i in range(len(label)):
		mm.append(str(label[i])+"-"+str(y_train[i]))
	print(np.unique(np.array(mm)))
	table1=np.unique(label)
	table2=np.unique(y_train)
	x_test = file_out.iloc[0:sizetest, 0:osize].values
	# x_test = torch.tensor(x_test, dtype=torch.float32).numpy()
	y_test = file_out.iloc[0:sizetest, osize].values
	for n in range(len(y_test)):
		if y_test[n]=="sittting" :
			y_test[n]="sitting"
	print(np.unique(y_test),111)
	z=[]
	for k in y_test:
		z.append(d[str(k)])
	y_test=np.array(z)
	# if np.unique(y_test).shape[0]!=np.unique(y_train).shape[0]:
	# 	for i in range(y_test.shape[0]):
	# 		y_test[i]=table2[ np.where( table1== y_test[i])[0][0]]
	# 	y_test=y_test.astype(np.int64)
	# else:
	# 	y_test=torch.as_tensor(le.fit_transform(y_test)).numpy()
	# print(y_train.dtype)
	# iris = datasets.load_iris()
	# x = iris.data
	# y = iris.target
	#
	# x = scale(x)
	#
	# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
	print(np.unique(y_train),np.unique(y_test))
	data = {'x_train': x_train, 
			'x_test': x_test, 
			'y_train': y_train, 
			'y_test': y_test}
	print("data in")
	return data
