import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from keras.datasets import mnist
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
from dataset import LGBSequence
import lightgbm
import time
x_train = LGBSequence( mode='train',xy="x")
y_train = x_train.getY()
x_test = LGBSequence(mode='val',xy="x")
y_test = x_test.getY()


# lgmodel = lgb.LGBMClassifier(
#     boosting_type='gbdt',
#     objective='multiclass',
#     learning_rate=0.01,
#     colsample_bytree=0.9,
#     subsample=0.8,
#     random_state=1,
#     n_estimators=100,
#     num_leaves=31)

# gbm = lightgbm.train(params,
#                 lgb_train,
#                 num_boost_round=20,
#                 valid_sets=lgb_eval,
#                 )

def lgb(n=10, c=0, sequence=1):
    model = LGBMClassifier(n_estimators=sequence)
    while(n):
        start=time.time()  
        # model.fit(x_test,y_test,verbose=1)
        # p=accuracy_score(y_test, model.predict(x_test))
        # print(p)
        print("time:",time.time()-start)        
        # model.booster_.save_model('lgb_model3.txt') 


        bst = lightgbm.Booster(model_file='lgb_model3.txt') 
        y=bst.predict(x_train)
        y=np.array(y)
        y=np.argmax(y,axis=-1)
        print(y)
        y_=np.array(y_train)
        print(y_)
        # p=accuracy_score(y_, y)
        p=np.mean(y==y_)
        print(p) 

if __name__ == '__main__':
    import time
    start=time.time()
    lgb()
    print("time:",time.time()-start)
