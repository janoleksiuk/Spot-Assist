import numpy as np
import vg

import read_data
import matplotlib.pyplot as plt
from pyMetaheuristic.algorithm import bat_algorithm, improved_grey_wolf_optimizer
from pyMetaheuristic.utils import graphs
from adjustText import adjust_text
from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
    precision_score, \
    f1_score, \
    recall_score
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tlabel = ['Collecting-0', 'bowing-1', 'cleaning-2', 'looking-3', 'opening-4',
          'passing-5', 'picking-6', 'placing-7', 'pushing-8', 'reading-9', 'sitting-10',
          'standing-11', 'standing_up-12', 'talking-13', 'turing_front-14',
          'turning-15', 'walking-16']


# Helper function that combines the pattern layer and summation layer
def gas(centre, x, sigma):
    centre = centre.reshape(1, -1)
    temp = -np.sum((centre - x) ** 2, axis=1)
    temp = temp / (2 * sigma * sigma)
    temp = np.exp(temp)
    gaussian = np.sum(temp)

    return gaussian


def mgas(centre, x, sigma):
    centre = centre.reshape(1, -1)
    temp = -np.linalg.norm(centre - x, ord=1, axis=1)
    temp = temp / (2 * sigma * sigma)
    temp = np.exp(temp)
    gaussian = np.sum(temp)

    return gaussian


def elaplas(centre, x, sigma):
    centre = centre.reshape(1, -1)
    temp = -np.sum((centre - x) ** 2, axis=1)
    temp = temp / sigma
    temp = np.exp(temp)
    gaussian = np.sum(temp)
    return gaussian


def laplas(centre, x, sigma):
    centre = centre.reshape(1, -1)
    temp = -np.linalg.norm(centre - x, ord=1, axis=1)
    # temp = -np.sum((centre - x) ** 2, axis=1)
    temp = temp / sigma
    temp = np.exp(temp)
    gaussian = np.sum(temp)
    print(gaussian)
    return gaussian


# def colaplas(centre, x, sigma):
#     centre = centre.reshape(1, -1)
#     num = np.dot([centre], np.array(x).T)  # 向量点乘
#     denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  # 求模长的乘积
#     res = num / denom
#     res[np.isneginf(res)] = 0
#     temp = res
#     temp = temp / sigma
#     temp = np.exp(temp)
#     gaussian = np.sum(temp)
#     return gaussian
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
    temp = temp / sigma
    temp = np.exp(-temp)
    gaussian = np.sum(temp)
    return gaussian


# def cosdistance(centre, x, sigma):
# 	centre = centre.reshape(1, -1)
# 	centre=centre.reshape(19,3)
# 	np.delete(centre,1,axis=0)
# 	x=x.reshape(x.shape[0],19,3)
# 	np.delete(x,1,axis=1)
# 	t=np.array([(np.dot([centre[i]], np.array(x[:,i]).T) )/(np.linalg.norm(centre[i]) * np.linalg.norm(x[:,i], axis=1))for i in range(centre.shape[0])])
# 	t=np.arccos(t)
# 	# l=np.pi-t
# 	# t=np.minimum(l,t)
# 	t=np.nan_to_num(t)
# 	t=np.sum(t,axis=0)
# 	res=t/18
# 	# res[np.isneginf(res)] = 0
# 	temp = res
# 	# for i in range (x.shape[0]):
# 	# 	temp[i]=centre.dot(x[i])/(np.linalg.norm(centre)*np.linalg.norm(x[i]))
# 	# temp = centre.dot(x)/(np.linalg.norm(centre)*np.linalg.norm(x))
# 	temp = temp / (2 * sigma * sigma)
# 	temp = np.exp(-temp)
# 	gaussian = np.sum(temp)
# 	return gaussian

def cosdistance(centre, x, sigma):
    centre = centre.reshape(1, -1)

    num = np.dot([centre], np.array(x).T)  # 向量点乘
    denom = np.linalg.norm(centre) * np.linalg.norm(x, axis=1)  # 求模长的乘积
    res = num / denom
    i = np.arccos(res)
    i = np.nan_to_num(i)
    tt = np.pi - i
    res = np.minimum(i, tt)
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


def PNN(data, sigma, tag):
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
    d = data['x_train'].shape[1]
    labels = np.unique(data['y_train'])
    num_class = len(labels)

    # Splits the training set into subsets where each subset contains data points from a particular class
    x_train_subsets = subset_by_class(data, labels)
    p = [len(i[0]) / data['x_train'].shape[0] for i in x_train_subsets]
    within = 0
    between = 0
    for n, subset in enumerate(x_train_subsets):
        within += p[n] * np.var(np.array(subset[0]))
        between += p[n] * np.sum((np.mean(np.array(subset[0]), axis=0) - np.mean(data['x_train'], axis=0)) ** 2)
    # Variable for storing the summation layer values from each class
    summation_layer = np.zeros(num_class)

    # Variable for storing the predictions for each test data point
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
    nm = np.unique(data['y_train'])
    print(nm, 11111)
    for i, test_point in enumerate(data['x_test']):
        print(i)
        joints = test_point.reshape([19, 3])
        for j, subset in enumerate(x_train_subsets):
            # Calculate summation layer
            # summation_layer[j] = np.sum(
            # 	laplas(test_point, subsepip install adjustTextt[0], sigma)) / (subset[0].shape[0]*2*pow(sigma,d)*between/within)
            if tag == 1:
                summation_layer[j] = np.sum(
                    gas(test_point, subset[0], sigma)) / (subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
            elif tag == 2:
                summation_layer[j] = np.sum(
                    mgas(test_point, subset[0], sigma)) / (subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
            elif tag == 3:
                summation_layer[j] = np.sum(
                    cosdistance(test_point, subset[0], sigma)) / (
                                             subset[0].shape[0] * pow(2 * np.pi, d / 2) * pow(sigma, d))
            elif tag == 4:
                summation_layer[j] = np.sum(
                    elaplas(test_point, subset[0], sigma)) / (subset[0].shape[0] * 2 * pow(sigma, d) * between / within)
            elif tag == 5:
                summation_layer[j] = np.sum(
                    laplas(test_point, subset[0], sigma)) / (subset[0].shape[0] * 2 * pow(sigma, d) * between / within)
            elif tag == 6:
                summation_layer[j] = np.sum(
                    colaplas(test_point, subset[0], sigma)) / (
                                             subset[0].shape[0] * 2 * pow(sigma, d) * between / within)

        # print(summation_layer, np.argmax(summation_layer),data['y_test'][i])
        for n in range(len(summation_layer)):
            summation_layer[n] = format(summation_layer[n], ".3e")
        # ax.cla()  # 清除键
        # ax.set_yticklabels(nm, fontsize=12)
        # # plt.xlim(0, 14)
        # # ax.set_xlim([0, 14])
        # ax.set_yticks(nm)
        # rects = ax.barh(nm, label='GroundTruth:' + str(data['y_test'][i]) + "---predict---" + str(np.argmax(summation_layer)),
        # 			   height=0.1, width=summation_layer)
        # ax.set_title('GroundTruth:' + str(tlabel[data['y_test'][i]]) + "---predict---" + str(tlabel[np.argmax(summation_layer)]))
        # for rect in rects:
        # 	ax.text(rect.get_width(), rect.get_y(), rect.get_width(), ha='left', va='bottom')
        #
        # fig.lines = []
        # for joint_connection in SkeletonConnectionMap:
        # 	endpoint_x = [joints[joint_connection[0]][0], joints[joint_connection[1]][0]]
        # 	endpoint_y = [joints[joint_connection[0]][1], joints[joint_connection[1]][1]]
        # 	endpoint_z = [joints[joint_connection[0]][2], joints[joint_connection[1]][2]]
        # 	fig.plot(endpoint_x, endpoint_y, endpoint_z, c='r')
        # plt.pause(1)
        # The index having the largest value in the summation_layer is stored as the prediction
        predictions[i] = np.argmax(summation_layer)

    return predictions


def print_metrics(y_test, predictions):
    print('Confusion Matrix')
    print(confusion_matrix(y_test, predictions))
    print('Accuracy: {}'.format(accuracy_score(y_test, predictions)))
    print('Precision: {}'.format(precision_score(y_test, predictions, average='macro')))
    print('Recall: {}'.format(recall_score(y_test, predictions, average='macro')))
    print('F1: {}'.format(f1_score(y_test, predictions, average='macro')))


# Target Function - It can be any function that needs to be minimize, However it has to have only one argument: 'variables_values'. This Argument must be a list of variables.
# For Instance, suppose that our Target Function is the Easom Function (With two variables x1 and x2. Global Minimum f(x1, x2) = -1 for, x1 = 3.14 and x2 = 3.14)

# Target Function: Easom Function

if __name__ == '__main__':
    data = read_data.input()


    #3-0.01866333
    #2-0.04229595
    #1-0.00745967
    #5-0.00346099
    #4-0.04273971
    #6-0.00127036
#Variables:  [0.00746044 0.        ]  Minimum Value Found:  -0.9437285223367697
#Variables:  [0.04129087 0.        ]  Minimum Value Found:  -0.9585481099656358
#Variables:  [0.01867524 0.        ]  Minimum Value Found:  -0.9506729667812142
#Variables:  [0.00011233 0.        ]  Minimum Value Found:  -0.9436569301260023
#Variables:  [0.00340949 0.        ]  Minimum Value Found:  -0.9585481099656358
#Variables:  [0.00069621 0.        ]  Minimum Value Found:  -0.9506729667812142
    def easom(variables_values=[0, 0]):
        x1, x2 = variables_values
        predictions = PNN(data, x1, 1)
        func_value = -accuracy_score(data['y_test'], predictions)-precision_score(data['y_test'], predictions, average='macro')-recall_score(data['y_test'], predictions, average='macro')
        return func_value
    def easom2(variables_values=[0, 0]):
        x1, x2 = variables_values
        predictions = PNN(data, x1, 2)
        func_value = -accuracy_score(data['y_test'], predictions)
        return func_value
    def easom3(variables_values=[0, 0]):
        x1, x2 = variables_values
        predictions = PNN(data, x1, 3)
        func_value = -accuracy_score(data['y_test'], predictions)-precision_score(data['y_test'], predictions, average='macro')-recall_score(data['y_test'], predictions, average='macro')
        return func_value
    def easom4(variables_values=[0, 0]):
        x1, x2 = variables_values
        predictions = PNN(data, x1, 4)
        func_value = -accuracy_score(data['y_test'], predictions)-precision_score(data['y_test'], predictions, average='macro')-recall_score(data['y_test'], predictions, average='macro')
        return func_value
    def easom5(variables_values=[0, 0]):
        x1, x2 = variables_values
        predictions = PNN(data, x1, 5)
        func_value = -accuracy_score(data['y_test'], predictions)
        return func_value
    def easom6(variables_values=[0, 0]):
        x1, x2 = variables_values
        predictions = PNN(data, x1, 6)
        func_value = -accuracy_score(data['y_test'], predictions)-precision_score(data['y_test'], predictions, average='macro')-recall_score(data['y_test'], predictions, average='macro')
        return func_value


    # iGWO - Parameters
    parameters = {
        'pack_size': 3,
        'min_values': (0, 0),
        'max_values': (1, 0),
        'iterations': 120,
        'verbose': True
    }
    igwo = improved_grey_wolf_optimizer(target_function=easom, **parameters)
    # igwo2 = improved_grey_wolf_optimizer(target_function=easom2, **parameters)
    igwo3 = improved_grey_wolf_optimizer(target_function=easom3, **parameters)
    igwo4 = improved_grey_wolf_optimizer(target_function=easom4, **parameters)
    # igwo5 = improved_grey_wolf_optimizer(target_function=easom5, **parameters)
    igwo6 = improved_grey_wolf_optimizer(target_function=easom6, **parameters)
    # BA - Solution
    print(igwo)
    variables = igwo[0][:-1]
    minimum = igwo[0][-1]
    print('Variables: ', variables, ' Minimum Value Found: ', minimum)
    # variables2 = igwo2[0][:-1]
    # minimum2 = igwo2[0][-1]
    # print('Variables: ', variables2, ' Minimum Value Found: ', minimum2)
    variables3 = igwo3[0][:-1]
    minimum3 = igwo3[0][-1]
    print('Variables: ', variables3, ' Minimum Value Found: ', minimum3)
    variables4 = igwo4[0][:-1]
    minimum4 = igwo4[0][-1]
    print('Variables: ', variables4, ' Minimum Value Found: ', minimum4)
    # variables5 = igwo5[0][:-1]
    # minimum5 = igwo5[0][-1]
    # print('Variables: ', variables5, ' Minimum Value Found: ', minimum5)
    variables6 = igwo6[0][:-1]
    minimum6 = igwo6[0][-1]
    print('Variables: ', variables6, ' Minimum Value Found: ', minimum6)
