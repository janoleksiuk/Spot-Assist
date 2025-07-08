from os.path import join
import torch
from sklearn import preprocessing
import numpy as np
import pandas as pd
import glob as gb

import matplotlib.pyplot as plt

# global variable
seed = 124
eps = 1e-15

d = {'Collecting': 0, 'bowing': 1, 'cleaning': 2,  'looking': 3, 'opening': 4,
		 'passing': 5, 'picking': 6, 'placing': 7, 'pushing': 8, 'reading': 9, 'sitting': 10, 'standing': 11,
		 'standing_up': 12, 'talking': 13, 'turing_front': 14, 'turning': 15, 'walking': 16}
# read all the csv files
def read_data(dataroot, file_ending='*.csv'):
    if file_ending is None:
        print("please specify file ending pattern for glob")
        exit()
    print(join(dataroot, file_ending))
    filenames = [i for i in gb.glob(join(dataroot, file_ending))]
    combined_csv = pd.concat([pd.read_csv(f, dtype=object) for f in filenames], sort=False,
                             ignore_index=True)  # dopisałem ignore_index=True
    return combined_csv


# read one csv file
def read_single_data(datafile):
    out_csv = pd.read_csv(datafile)
    return out_csv
def clean(data,label,index):
    data=data.to_numpy()
    label=label.to_numpy()
    # index=[]
    # for i in range(data.shape[0]):
    #     if data[i].any()==None:
    #         print("nan")
    #         index.append(i)
    data=np.delete(data,index,axis=0)
    label=np.delete(label,index,axis=0)
    return data,label
def normalization(M):
    M=M/np.sqrt(np.dot(M,M.T))
    return M
# load the csv files in data frame and start normalizing
def load_data_train(*params):
    # d = {"1.0": 0, "3.0": 1, "5.0": 2, "6.0": 3, "7.0": 4, "9.0": 5, "11.0": 6, "13.0": 7, "22.0": 8, "25.0": 9, "28.0": 10,
    #      "32.0": 11, "33.0": 12, "34.0": 13, "35.0": 14, "42.0": 15, "44.0": 16, "47.0": 17, "49.0": 18, "51.0": 19}
    # d = {"1.0": 0, "2.0": 1, "4.0": 2, "5.0": 3, "8.0": 4, "9.0": 5, "13.0": 6, "14.0": 7, "15.0": 8, "16.0": 9,
    #      "17.0": 10,
    #      "18.0": 11, "19.0": 12, "25.0": 13, "27.0": 14, "28.0": 15, "31.0": 16, "34.0": 17, "35.0": 18, "38.0": 19}
    d = {'Collecting': 0, 'bowing': 1, 'cleaning': 2, 'drinking': 3, 'eating': 4, 'looking': 5, 'opening': 6,
         'passing': 7, 'picking': 8, 'placing': 9, 'pushing': 10, 'reading': 11, 'sitting': 12, 'standing': 13,
         'standing_up': 14, 'talking': 15, 'turing_front': 16, 'turning': 17, 'walking': 18}
    # dataroot = ".../train_data/pixel/"
    dataroot = params[0]
    idx=params[1]
    data_path = read_data(dataroot, '*.csv')
    num_records, num_features = data_path.shape
    print("there are {} flow records with {} feature dimension".format(num_records, num_features))
    # there is white spaces in columns names e.g. ' Destination Port'
    # So strip the whitespace from  column names
    data = data_path.rename(columns=lambda x: x.strip())
    print('stripped column names')
    print('dropped bad columns')
    data.replace([np.inf, -np.inf], np.nan).dropna()
    num_records, num_features = data.shape
    print("there are new {} flow records with {} feature dimension".format(num_records, num_features))
    data = data.drop(data[data.label =="not specified"].index)
    data = data.sample(frac=1.0)  # 打乱所有数据
    data = data.reset_index(drop=True)
    # if idx==1:
    #     # data=pd.read_csv("weight/train-all-correct-2.csv")
    #     data = pd.read_csv(
    #         '/mnt/storage/buildwin/desk_backword/sota/DIVA-master/dataset/wut/pku_train_new20_filted.csv')
    #     # data = pd.read_csv(
    #     #     '/mnt/storage/buildwin/desk_backword/sota/skeleton-action-recognition-master/data/ntu/xview/train_data_train.csv')
    #
    # else:
    #     # data = pd.read_csv("weight/test-all-correct-2.csv")
    #     data = pd.read_csv( '/mnt/storage/buildwin/desk_backword/pku mmd/video-tets/data_all.csv')
    if idx == 1:
        data.to_csv("train-all.csv",index=False)
    else:
        data.to_csv("test-all.csv",index=False)
    label=data.iloc[:,-1]
    data=data.iloc[:,:-1]
    # data,label=clean(data,label)
    data = data.astype('float64')
    nan_count = data.isnull().sum().sum()
    if nan_count > 0:
        print("nan")
        data.fillna(data.mean(), inplace=True)
        # print('filled NAN')
        # Normalising all numerical features:
        # cols_to_norm = list(data.columns.values)[:68]
        # print('cols_to_norm:\n', cols_to_norm)
    data = data.astype(np.float32)
    data = data.astype(float).apply(pd.to_numeric)
    for i, r in data.iterrows():
        t = normalization(r)
        data.at[i] = t
    # data=data.to_numpy()
    data,label=clean(data,label,data[data.isnull().T.any()].index.to_numpy())
    # print(data[data.isnull().T.any()].index.to_numpy())
    # le = preprocessing.LabelEncoder()
    print(np.unique(label))
    # label=label.astype(np.float)
    # label = label.astype(np.int)
    # print(np.unique(label))
    # le = preprocessing.LabelEncoder()
    # label2 = torch.as_tensor(end.fit_transform(label))
    label2 = []
    for i in label:
        label2.append(d[str(i)])
    label2 = torch.from_numpy(np.array(label2).astype(np.float32)).clone()
    # label2=torch.from_numpy(label.to_numpy())
    # y_train = torch.as_tensor(le.fit_transform(label)).numpy()
    #print('There are {} nan entries'.format(nan_count))
    print(np.unique(label2.to('cpu').detach().numpy().copy()))
    # if idx == 1:
    #     data=np.load("weight2/wut-train-z.npy")
    #     data=pd.DataFrame(data)
    #     label2=np.load("weight2/wut-train-y.npy")
    # else:
    #     data = np.load("weight2/wut-test-z.npy")
    #     data = pd.DataFrame(data)
    #     label2 = np.load("weight2/wut-test-y.npy")
    # label2=torch.from_numpy(label2)
    # print(label2[:50],label[:50])
    print(label2.unique(),"label2")
    data = data.reshape((data.shape[0],  57))
    # data = torch.from_numpy(data.astype(np.float32)).clone()
    #data-----DataFrame
    # print('converted to numeric\n', data)
    # data.hist(figsize=(3,5))
    # plt.show()
    return data,label2,label


# load the csv files from test folder in data frame and start normalizing
def load_data_test(*params):
    # d = {"1": 0, "3": 1, "5": 2, "6": 3, "7": 4, "9": 5, "11": 6, "13": 7, "22": 8, "25": 9, "28": 10,
    #      "32": 11, "33": 12, "34": 13, "35": 14, "42": 15, "44": 16, "47": 17, "49": 18, "51": 19}
    # d = {'Collecting': 0, 'cleaning': 1, 'looking': 2, 'opening': 3, 'passing': 4, 'picking': 5, 'placing': 6,
    #      'pushing': 7, 'reading': 8, 'sitting': 9, 'standing': 10, 'talking': 11, 'turning': 12, 'walking': 13}
    d={'Collecting':0 ,'bowing':1 ,'cleaning':2, 'drinking':3, 'eating':4, 'looking':5, 'opening':6,
 'passing':7 ,'picking':8 ,'placing':9, 'pushing':10, 'reading':11 ,'sitting' :12,'standing':13,
 'standing_up':14, 'talking':15 ,'turing_front':16 ,'turning':17, 'walking':18}
    dataroot = params[0]
    end = params[1]
    # # dataroot = "./correct_data/train/"
    data_path = read_data(dataroot, '*.csv')
    num_records, num_features = data_path.shape
    print("there are {} flow records with {} feature dimension".format(num_records, num_features))
    # there is white spaces in columns names e.g. ' Destination Port'
    # So strip the whitespace from  column names
    data = data_path.rename(columns=lambda x: x.strip())
    print('stripped column names')
    print('dropped bad columns')
    num_records, num_features = data.shape
    print("there are new {} flow records with {} feature dimension".format(num_records, num_features))
    print(data)
    data = data.drop(data[data.label =="not specified"].index)
    label = data.iloc[:, -1]
    data = data.iloc[:, :-1]
    print(data.shape, label.shape)
    data = data.astype('float64')
    label = label.to_numpy()
    label2=[]
    for i in label:
        label2.append(0)
    label2 = torch.from_numpy(np.array(label2).astype(np.float32)).clone()


    data = data.astype(np.float32)
    data = data.astype(float).apply(pd.to_numeric)
    for i, r in data.iterrows():
        r = normalization(r)
        data.at[i] = r
    data = data.to_numpy()
    data = data.reshape((data.shape[0], 57))
    data = torch.from_numpy(data.astype(np.float32)).clone()

    return data, label2, label
