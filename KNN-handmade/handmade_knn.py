import numpy as np
import pandas as pd
import sklearn.datasets 
import collections

wine = sklearn.datasets.load_wine(as_frame=True)
#print(wine.DESCR)
wine = wine.frame
#print(wine.shape)

def split_function(data,split_ratio):
    shuffled = data.sample(frac=1,random_state=42)
    train= shuffled[:int(len(shuffled)*split_ratio)]
    test = shuffled[int(len(shuffled)*split_ratio):]
    x_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    x_test = test.iloc[:,:-1]
    y_test = test.iloc[:,-1]
    return x_train,y_train,x_test,y_test

x_tr,y_tr,x_ts,y_ts = split_function(wine,0.8)

def euclidean_distance(point_p,point_q):
    sum_of_squares = 0
    if len(point_p) == len(point_q):
        for i in range(0,len(point_p)):
            sum_of_squares = sum_of_squares + pow((point_q[i]-point_p[i]),2)
        e_dist = round(np.sqrt(sum_of_squares),2)
        return e_dist
    else:
         raise ValueError("Points coordinates numbers are not equal")



def majority_vote(labels):
    cnt = collections.Counter(labels)
    winner = cnt.most_common(1)[0][0]
    return winner


def predict(X_train,Y_train,X_test,k):
    distances = []
    for r in range(0,len(X_train)):
        distance = (euclidean_distance(list(X_train.iloc[r]),list(X_test)),Y_train.iloc[r])
        distances.append(distance)
    distances = sorted(distances,key=lambda x: x[0])
    distances = distances[0:k]
    labels = [d[1] for d in distances]
    winning_label = majority_vote(labels)
    return winning_label
        


X_ts_1 = x_ts.iloc[0]
# print(X_t_1)
x_ts_1_actual_label = y_ts.iloc[0]
# print(x_t_1_label)
x_ts_1_predicted_label = predict(x_tr,y_tr,X_ts_1,3)
print(f"Predicted label: {x_ts_1_predicted_label}")
print(f"Actual label: {x_ts_1_actual_label}")

