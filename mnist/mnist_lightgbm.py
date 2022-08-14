import numpy as np
from time import process_time
import matplotlib.pyplot as plt
from keras.datasets import mnist
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss


def lgb(n=10, c=0, sequence=1):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    RESHAPED = 784
    x_train = x_train.reshape(60000, RESHAPED)
    x_test = x_test.reshape(10000, RESHAPED)
    x_train = x_train.astype('float64')
    x_test = x_test.astype('float64')
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    proba_test = np.zeros((n, y_test.shape[0], len(np.unique(y_test))))
    proba_train = np.zeros((n, y_train.shape[0], len(np.unique(y_train))))

    test_score = []
    train_score = []

    tr_time = []
    seq = []
    while(n):

        model = LGBMClassifier(n_estimators=sequence)

        t0 = process_time()
        model.fit(x_train, y_train)
        tr_time.append(process_time() - t0)
        test_score.append(accuracy_score(y_test, model.predict(x_test)))
        train_score.append(accuracy_score(y_train, model.predict(x_train)))
        proba_test[c, ] = model.predict_proba(x_test)
        proba_train[c, ] = model.predict_proba(x_train)

        seq.append(sequence)
        sequence *= 2
        n -= 1
        c += 1

        ce_train = []
        ce_test = []

    for i in range(10):
        ce_test.append(log_loss(y_test, proba_test[i]))
        ce_train.append(log_loss(y_train, proba_train[i]))
        
        np.savetxt('round'+ str(i) + 'proba_test.csv', proba_test[i])
        np.savetxt('round'+ str(i) + 'proba_train.csv', proba_train[i])

    np.savetxt('test_score.csv', test_score, delimiter=',')
    np.savetxt('train_score.csv', train_score, delimiter=',')

    np.savetxt('ce_test.csv', ce_test, delimiter=',')
    np.savetxt('ce_train.csv', ce_train, delimiter=',')

    
    fig, ax1 = plt.subplots()

    l1 = ax1.plot(seq, ce_test, ':', label='loss-test', color='r')
    l2 = ax1.plot(seq, ce_train, ':', label='loss-train', color='b')
    ax1.set_xlabel("Boost rounds")
    ax1.set_ylabel("Cross Entropy")

    ax2 = ax1.twinx()
    l3 = ax2.plot(seq, test_score, label='accuracy-test', color='r')
    l4 = ax2.plot(seq, train_score, label='accuracy-train', color='b')
    ax2.set_ylabel("Accuracy")

    lb = l1 + l2 + l3 + l4
    label = [l.get_label() for l in lb]

    ax1.legend(lb, label, loc=0)
    plt.title('loss curve/ accuracy')
    plt.savefig('loss_curve.jpg', dpi=500)


if __name__ == '__main__':
    import time
    start=time.time()
    lgb()
    print("time:",time.time()-start)
