import threading
import numpy as np
import read_data
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import os
import sys
import signal
from pathlib import Path


MODEL_PATH = Path("pose-classifier") / "reference_data" / "reference_data.csv"

# CUDA setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Poses
dic = {'sitting': 0, 'standing': 1, 'sitting_1hand': 2, 'standing_1hand': 3}


def assemble_dir(*parts: str) -> str:
    project_root = Path(__file__).resolve().parent.parent  
    full_path = project_root.joinpath(*parts)
    return str(full_path)

# Helper functions that combines the pattern layer and summation layer
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
	temp = temp /  sigma
	temp = np.exp(temp)
	gaussian = np.sum(temp)
	return gaussian


def colaplas(centre, x, sigma):
	centre = centre.reshape(1, -1)
	num = np.dot([centre], np.array(x).T)  
	denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  
	try: 
		res = num / denom
	except Exception as e:
		print(f"[CLASSIFIER]: Error {e} in colaplas function determining. Returning 0")
		return 0 
	
	i = np.arccos(res)
	i = np.nan_to_num(i)
	tt = np.pi - i
	res = np.minimum(i, tt)
	temp = res
	temp = temp/sigma
	temp = np.exp(-temp)
	gaussian = np.sum(temp)
	return gaussian


def cosdistance(centre, x, sigma):
	centre = centre.reshape(1, -1)
	num = np.dot([centre], np.array(x).T)  
	denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  
	try: 
		res = num / denom
	except Exception as e:
		print(f"[CLASSIFIER]: Error {e} in cosdistance function determining. Returning 0")
		return 0 
	
	i = np.arccos(res)
	i = np.nan_to_num(i)
	i = i[0][0]
	tt = np.pi-i
	res = np.minimum(i,tt)
	temp = res
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


def handle_prediction(predictions, shm):
	# find dominant array value
	values, counts = np.unique(np.array(predictions), return_counts=True)
	value =  int(values[np.argmax(counts)])

	# write pose value to the endpoint
	shm.buf[:8] = value.to_bytes(8, byteorder='little', signed=True)
	return value


# neural network body
def PNN(data,sigma,tag):
	num_testset = data['x_test'].shape[0]
	d = data['x_train'].shape[1]
	labels = np.unique(data['y_train'])
	num_class = len(labels)

	# Splits the training set into subsets where each subset contains data points from a particular class	
	x_train_subsets = subset_by_class(data, labels)	
	p = [len(i[0])/data['x_train'].shape[0] for i in x_train_subsets]
	within = 0
	between = 0
	for n, subset in enumerate(x_train_subsets):
		within += p[n]*np.var(np.array(subset[0]))
		between += p[n]*np.sum((np.mean(np.array(subset[0]),axis=0)-np.mean(data['x_train'],axis=0))**2)
	summation_layer = np.zeros(num_class)
	predictions = np.zeros(num_testset)

	# NN layers 
	for i, test_point in enumerate(data['x_test']): 
		for j, subset in enumerate(x_train_subsets):
			if tag == 1:
				summation_layer[j] = np.sum(
				gas(test_point, subset[0], sigma)) / (subset[0].shape[0] *pow(2*np.pi, d/2)* pow(sigma,d))
			elif tag == 2:
				summation_layer[j] = np.sum(
					mgas(test_point, subset[0], sigma)) / (subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
			elif tag == 3:
				summation_layer[j] = np.sum(
					cosdistance(test_point, subset[0], sigma)) / (subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
			elif tag == 4:
				summation_layer[j] = np.sum(
			elaplas(test_point, subset[0], sigma)) / (subset[0].shape[0]*2*pow(sigma,d)*between/within)
			elif tag == 5:
				summation_layer[j] = np.sum(
					laplas(test_point, subset[0], sigma)) / (subset[0].shape[0] * 2 * pow(sigma, d) *between / within)
			elif tag == 6:
				summation_layer[j] = np.sum(
					colaplas(test_point, subset[0], sigma)) / (subset[0].shape[0] * 2 * pow(sigma, d) * between / within)

		for  n in range(len(summation_layer)):
			summation_layer[n] = format(summation_layer[n],".3e")
		predictions[i] = np.argmax(summation_layer)
	return predictions

	
def main(argv):
	# multiprocessing shared memory segments
	shm_detected_pose_name = argv[1]
	shm_detected_pose = shared_memory.SharedMemory(name=shm_detected_pose_name)
	shm_pnn_input_name = argv[2]
	shm_pnn_input = shared_memory.SharedMemory(name=shm_pnn_input_name)

	# handling process termination
	def cleanup(signum=None, frame=None):
		print("[--- CLASSIFIER ---]: cleaning up shared memory...")
		shm_detected_pose.close()
		shm_pnn_input.close()
		exit(0)
	signal.signal(signal.SIGTERM, cleanup)
	signal.signal(signal.SIGINT, cleanup)

	# import reference data to rapid determine mocel weights
	model_dir = assemble_dir("pose-classifier", "reference_data", "reference_data.csv")
	data1, _ = read_data.input(model_dir, isTrain=True, isCSV=True)
	
	# prediction loop
	while True:
		pnn_input = np.ndarray((15, 57), dtype=np.float64, buffer=shm_pnn_input.buf)
		try:
			data2, read_data_single_exit_code = read_data.input(pnn_input, isTrain=False)
		except Exception as e:
			print(f"[--- CLASSIFIER ---]: Error {e}. Incorrect pnn memory sharing input.")
			continue

		if read_data_single_exit_code == 1:
			#rearranging arrays
			ordered_keys = ['x_train', 'x_test', 'y_train', 'y_test']
			combined = {**data1, **data2}
			data = {k: combined[k] for k in ordered_keys}
			
			predictions=PNN(data, 0.01867524, 3)
			handle_prediction(predictions=predictions, shm=shm_detected_pose)
		else:
			print("[--- CLASSIFIER ---]: Corrupted data - prediciton skipped")
			continue


if __name__ == '__main__':
	try:
		main(sys.argv)
	except Exception as e:
		print(f"[--- CLASSIFIER ---]: {e}. Error exit.")