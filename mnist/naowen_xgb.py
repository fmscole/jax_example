import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from keras.datasets import mnist
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from dataset import LGBSequence
import xgboost as xgb
import time
x_train = LGBSequence( mode='train',xy="x")
y_train = x_train.getY()
x_test = LGBSequence(mode='val',xy="x")
y_test = x_test.getY()




def lgb(n=10, c=0, sequence=1):
    model = XGBClassifier(n_estimators=sequence)
    while(n):
        start=time.time()  
        # model.fit(x_test,y_test)
        # p=accuracy_score(y_test, model.predict(x_test))
        # print(p)
        print("time:",time.time()-start)        
        # model.save_model('xgb_model.txt') 


        bst = xgb.Booster(model_file='xgb_model.txt') 
        y=bst.predict(x_test)
        y=np.array(y)
        y=np.argmax(y,axis=-1)
        print(y)
        y_=np.array(y_test)
        print(y_)
        # p=accuracy_score(y_, y)
        p=np.mean(y==y_)
        print(p) 

if __name__ == '__main__':
    import time
    start=time.time()
    lgb()
    print("time:",time.time()-start)
