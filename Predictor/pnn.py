import threading
import numpy as np
import read_data
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, \
							confusion_matrix, \
							precision_score, \
							f1_score, \
							recall_score
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# tlabel=['Collecting-0' ,'bowing-1' ,'cleaning-2', 'looking-3', 'opening-4',
#  'passing-5' ,'picking-6', 'placing-7', 'pushing-8', 'reading-9' , 'sitting-10',
#  'standing-11' ,'standing_up-12', 'talking-13' ,'turing_front-14',
#  'turning-15' ,'walking-16']
# Helper function that combines the pattern layer and summation layer
dic = {'sitting': 0, 'standing': 1, 'sitting_1hand': 2, 'standing_1hand': 3}
def gas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	temp = -np.sum((centre - x) ** 2, axis = 1)
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(temp)
	gaussian = np.sum(temp)

	return gaussian


def mgas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	temp = -np.linalg.norm(centre-x,ord=1,axis=1)
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(temp)
	gaussian = np.sum(temp)

	return gaussian
	
def elaplas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	temp = -np.sum((centre - x) ** 2, axis=1)
	temp = temp /  sigma
	temp = np.exp(temp)
	gaussian = np.sum(temp)
	return gaussian

def laplas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	temp = -np.linalg.norm(centre-x,ord=1,axis=1)
	# temp = -np.sum((centre - x) ** 2, axis=1)
	temp = temp /  sigma
	temp = np.exp(temp)
	gaussian = np.sum(temp)
	return gaussian

def colaplas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	num = np.dot([centre], np.array(x).T)  # 向量点乘
	denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  # 求模长的乘积
	res = num / denom
	i = np.arccos(res)
	i = np.nan_to_num(i)
	tt = np.pi - i
	res = np.minimum(i, tt)
	# res[np.isneginf(res)] = 0
	# res[np.isneginf(res)] = 0
	temp = res
	temp = temp/sigma
	temp = np.exp(-temp)
	gaussian = np.sum(temp)
	return gaussian

def cosdistance(centre, x, sigma):
	centre = centre.reshape(1, -1)

	num = np.dot([centre], np.array(x).T)  # 向量点乘

	denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  # 求模长的乘积
	res = num / denom
	# for i in res[0][0]:
	# 	if i <0:
	# 		print(i)
	# 		print(np.arccos(i))
	i=np.arccos(res)
	i = np.nan_to_num(i)
	i=i[0][0]
	tt=np.pi-i
	res=np.minimum(i,tt)
	# res[np.isneginf(res)] = 0
	temp = res
	# for i in range (x.shape[0]):
	# 	temp[i]=centre.dot(x[i])/(np.linalg.norm(centre)*np.linalg.norm(x[i]))
	# temp = centre.dot(x)/(np.linalg.norm(centre)*np.linalg.norm(x))
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(-temp)
	gaussian = np.sum(temp)
	return gaussian

def subset_by_class(data, labels):

	x_train_subsets = []
	
	for l in labels:
		indices = np.where(data['y_train'] == l)
		x_train_subsets.append(data['x_train'][indices, :])

	return x_train_subsets

def action_task(label):
	# print("the action---"+str(label)+"---is done")
	# print("the robot command can put here!")
	pass

# this is function outputting predicition - returns value of pose which will be passed to robot controller
def handle_prediction(predictions, endpoint_path):

	#find dominant array value
	values, counts = np.unique(np.array(predictions), return_counts=True)
	value =  int(values[np.argmax(counts)])
	pose  = [k for k, v in dic.items() if v == value][0]
	# print(pose)
	
	#write pose value to the txt endpoint file
	with open(endpoint_path, 'w') as f:
		f.write(str(value))

	#ADDITIONALLY writing pose string to another txt endpoint as informative feedback
	try:
		with open(r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\pose_string.txt', 'w') as ff:
			ff.write(str(pose))
	except Exception as e:
		pass

	return value

#PNN implementation
def PNN(data,sigma,tag):
	SkeletonConnectionMap = [[1, 0],
							 [2, 1],
							 [3, 2],
							 [4, 2],
							 [5, 4],
							 [6, 5],
							 [7, 6],
							 [8, 2],
							 [9, 8],
							 [10, 9],
							 [11, 10],
							 [12, 0],
							 [13, 12],
							 [14, 13],
							 [15, 0],
							 [16, 15],
							 [17, 16],
							 [18, 3],
							 ]
	num_testset = data['x_test'].shape[0]
	d=data['x_train'].shape[1]
	labels = np.unique(data['y_train'])
	num_class = len(labels)
	# Splits the training set into subsets where each subset contains data points from a particular class	
	x_train_subsets = subset_by_class(data, labels)	
	p=[len(i[0])/data['x_train'].shape[0] for i in x_train_subsets]
	within=0
	between=0
	for n, subset in enumerate(x_train_subsets):
		within+=p[n]*np.var(np.array(subset[0]))
		between+=p[n]*np.sum((np.mean(np.array(subset[0]),axis=0)-np.mean(data['x_train'],axis=0))**2)
	# Variable for storing the summation layer values from each class
	summation_layer = np.zeros(num_class)
	
	# Variable for storing the predictions formacro each test data point
	predictions = np.zeros(num_testset)
	ax = plt.subplot(1, 2, 1)

	fig = plt.subplot(1, 2, 2, projection='3d')
	fig.view_init(-90, 90)
	fig.set_xlabel('x')
	fig.set_ylabel('y')
	fig.set_zlabel('z')
	fig.set_xlim(-1.500, 1.500)
	fig.set_ylim(-1.000, 1.000)
	fig.set_zlim(-1.000, 1.000)
	nm=np.unique(data['y_train'])
	for i, test_point in enumerate(data['x_test']):
		joints = test_point.reshape([19, 3])
		for j, subset in enumerate(x_train_subsets):
			if tag==1:
				summation_layer[j] = np.sum(
				gas(test_point, subset[0], sigma)) / (subset[0].shape[0] *pow(2*np.pi, d/2)* pow(sigma,d))
			elif tag==2:
				summation_layer[j] = np.sum(
					mgas(test_point, subset[0], sigma)) / (subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
			elif tag==3:
				summation_layer[j] = np.sum(
					cosdistance(test_point, subset[0], sigma)) / (subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
			elif tag==4:
				summation_layer[j] = np.sum(
			elaplas(test_point, subset[0], sigma)) / (subset[0].shape[0]*2*pow(sigma,d)*between/within)
			elif tag==5:
				summation_layer[j] = np.sum(
					laplas(test_point, subset[0], sigma)) / (subset[0].shape[0] * 2 * pow(sigma, d) *between / within)
			elif tag==6:
				summation_layer[j] = np.sum(
					colaplas(test_point, subset[0], sigma)) / (subset[0].shape[0] * 2 * pow(sigma, d) * between / within)

		# print(summation_layer, np.argmax(summation_layer),data['y_test'][i])
		for  n in range(len(summation_layer)):
			summation_layer[n]=format(summation_layer[n],".3e")
		predictions[i] = np.argmax(summation_layer)
		# print([k for k,v in dic.items() if v==predictions[i]])
		if i==0:
			continue
		elif i==predictions.shape[0]-1:
			thread = threading.Thread(target=action_task, args=([k for k, v in dic.items() if v == predictions[i - 1]]))
			thread.start()
			# time.sleep(5)
		else:
			if predictions[i]!=predictions[i-1]:
				thread=threading.Thread(target=action_task,args=([k for k,v in dic.items() if v==predictions[i-1]]))
				thread.start()
				# time.sleep(5)

	return predictions

def print_metrics(y_test, predictions):
	predictions=predictions.astype(int)
	print('Confusion Matrix')
	print(y_test,predictions)
	# print(confusion_matrix(y_test, predictions))
	print('Precision: {}'.format(precision_score(y_test, predictions, average = 'macro')))
	print('Recall: {}'.format(recall_score(y_test, predictions, average = 'macro')))
	print('F1: {}'.format(f1_score(y_test, predictions, average='macro')))
	
if __name__ == '__main__':

	data1, _ = read_data.input(trainpath = r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\trainset_100425_v2.csv', isTrain= True)
	
	while True:
		data2, read_data_single_exit_code = read_data.input(trainpath = r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\19.csv', isTrain = False)

		if read_data_single_exit_code == 1:

			#rearranging arrays
			ordered_keys = ['x_train', 'x_test', 'y_train', 'y_test']
			combined = {**data1, **data2}
			data = {k: combined[k] for k in ordered_keys}
			
			#predicitng
			predictions=PNN(data, 0.01867524 , 3)

			#handling predictions
			value = handle_prediction(predictions=predictions, endpoint_path=r'C:\Users\j.oleksiuk_ladm\Desktop\Spot Ecosystem\prod\behaviour_code.txt')
		
		else:
			print("Corrupted data - prediciton skipped")
			continue

	