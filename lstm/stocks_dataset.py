import numpy as np
import pandas as pd


data = pd.read_csv('IBM.csv')
all_data = data.iloc[:, 1:7]
all_data=all_data.to_numpy()
data_max=np.max(all_data,axis=0)
data_min=np.min(all_data,axis=0)
all_data_scaled=(all_data-data_min)/(data_max-data_min)
# print(all_data_scaled)
# sc = MinMaxScaler(feature_range = (0, 1),)
# all_data_scaled = sc.fit_transform(all_data)

features = []
labels = []
for i in range(60, len(all_data_scaled)):
    features.append(all_data_scaled[i-60:i, ])
    labels.append(all_data_scaled[i, ])
features, labels = np.array(features), np.array(labels)
features = np.reshape(features, (features.shape[0], features.shape[1], -1))
x_train,x_test,y_train,y_test= features[:6500],features[6500 : 7200],labels[:6500],labels[6500 :7200]


print(len(all_data_scaled))
print(y_train.shape,y_test.shape)
'''
(7218, 60, 6)
(7218, 60, 6)
shape of x_train: (6550, 60, 6)
shape of x_val: (364, 60, 6)
shape of x_test: (304, 60, 6)
shape of y_train: (6550,)
shape of y_val: (364,)
shape of y_test: (304,)
'''