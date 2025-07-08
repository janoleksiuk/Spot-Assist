import numpy as np
import copy
import numpy as np
import time
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
tlabel=['Collecting' ,'bowing' ,'cleaning', 'looking', 'opening',
 'passing' ,'picking', 'placing', 'pushing', 'reading' , 'sitting',
 'standing' ,'standing_up', 'talking' ,'turing_front',
 'turning' ,'walking']
groundtruth=np.load("/home/hexin/桌面/new_warsadata/exampletest/groundtruth.npy", allow_pickle=True
)
colors=[]
for i in range(max(groundtruth)+1):
    # time.sleep(0.5)
    colors.append( (np.random.random(), np.random.random(), np.random.random()))
global lgcolor
global lgaction
lgcolor=[]
lgaction=[]
knn=np.load("/home/hexin/桌面/new_warsadata/exampletest/knn1.npy",allow_pickle=True)
cat=np.load("/home/hexin/桌面/new_warsadata/exampletest/cat.npy",allow_pickle=True)
xbnet=np.load("/home/hexin/桌面/new_warsadata/exampletest/xbnet.npy",allow_pickle=True)
al=np.load("/home/hexin/桌面/new_warsadata/exampletest/ang-L.npy",allow_pickle=True)
el=np.load("/home/hexin/桌面/new_warsadata/exampletest/E-L.npy",allow_pickle=True)
eg=np.load("/home/hexin/桌面/new_warsadata/exampletest/E-G.npy",allow_pickle=True)
ag=np.load("/home/hexin/桌面/new_warsadata/exampletest/ang-G.npy",allow_pickle=True)
ax = plt.subplot()
def plts(groundtruth,index):
    global lgcolor
    global lgaction
    i = 0
    j = 0
    set = []
    while j < len(groundtruth) - 1:
        j = j + 1
        if groundtruth[j] != groundtruth[i]:
            set.append([groundtruth[i], i, j])
            i = j
    set.append([groundtruth[i], i, j])
    for i in range(len(set)):
        if i == 0:
            plt.barh(np.array([index]), set[i][2], align="center", edgecolor="black", color=colors[int(set[i][0])],
                     label="orgin")
            if colors[int(set[i][0])] not in lgcolor:
                lgcolor.append(colors[int(set[i][0])])
            if tlabel[int(set[i][0])] not in lgaction:
                lgaction.append(tlabel[int(set[i][0])])
        else:
            plt.barh(np.array([index]), set[i][2] - set[i][1], align="center", edgecolor="black", color=colors[int(set[i][0])],
                     left=set[i][1], label="change")
            if colors[int(set[i][0])] not in lgcolor:
                lgcolor.append(colors[int(set[i][0])])
            if tlabel[int(set[i][0])] not in lgaction:
                lgaction.append(tlabel[int(set[i][0])])
plts(groundtruth,14)
plts(al,6)
plts(el,8)
plts(eg,12)
plts(ag,10)
plts(xbnet,4)
plts(knn,2)
plts(cat,0)
plt.xlabel('Frame number')
ax.set_yticklabels(("0","CatBoost","KNN","XBNET","Ang-Laplace","Euc-Laplace","Ang-guassian","Euc-guassian","Groundtruth"), minor=False)
patches = [ mpatches.Patch(color=lgcolor[i], label="{:s}".format(lgaction[i]) ) for i in range(len(lgcolor)) ]
ax=plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
#下面一行中bbox_to_anchor指定了legend的位置
ax.legend(handles=patches, bbox_to_anchor=(0.7,1.12), ncol=4) #生成legend

plt.show()
