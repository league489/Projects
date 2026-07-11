# KNN algorithm  - my own implementation

This is my implementation of KNN algorithm, build from scratch.

## What is KNN?
kNN (K - Nearest Neighbours) is a supervised learning algorithm used for classification. It classifies a test point based on the majority label of its k nearest neighbors in the training set.

## Requirements 
- numpy
- pandas
- sklearn.datasets
- collections
## Helper functions
- *split_function(data,split_ratio)*:  
Function that splits **data** dataset by **split_ratio** parameter. Function returns following splits: X_train, Y_train, X_test, Y_test.
- *euclidean_distance(point_p,point_q)*:  
Function that computates and returns Euclidean distance metric for given **point_p** and **point_q** points. Function performs length equality check before computation, raising ValueError exception if the points feature numbers don't match.
- *majority_vote(labels)*:  
Function that counts **labels** values with *collections.Counter()*, returning first most common label.
## Main function
*predict(X_train,Y_train,X_test,k)*:  
Function that computates Euclidian distance metric between **X_train** points and **X_test** point, selects **k** nearest neighbours and perfroms majority vote at their labels. Function returns winning label for **X_test** point.
## Examples
The algorithm is demonstrated on the Wine dataset (3 classes, 13 features)
### Single point test
``` python
X_ts_1 = x_ts.iloc[0]
# print(X_t_1)
x_ts_1_actual_label = y_ts.iloc[0]
# print(x_t_1_label)
x_ts_1_predicted_label = predict(x_tr,y_tr,X_ts_1,3)
```
Result
``` bash
Predicted label: 2
Actual label: 2
```
### Accuracy test for all points in test split

``` python
match = 0
for point in range(0,len(x_ts)):
    predicted_label = predict(x_tr,y_tr,x_ts.iloc[point],3)
    actual_label = y_ts.iloc[point]
    if predicted_label == actual_label:
        match += 1
accuracy = round(match/len(x_ts), 2)
print(f"Accuracy: {accuracy}")
```
Result
``` bash
Accuracy: 0.83
```
Accuracy comparsion with sklearn KNN implementation
``` python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

sk_model = KNeighborsClassifier(n_neighbors=3)
sk_model.fit(x_tr, y_tr)
sk_pred = sk_model.predict(x_ts)
print(f"sklearn accuracy: {accuracy_score(y_ts, sk_pred):2f}")
```
Result
``` bash
sklearn accuracy: 0.81
```
## How to run
```python
# Load data
from sklearn.datasets import load_wine
data = load_wine(as_frame=True).frame
# Split
x_tr, y_tr, x_ts, y_ts = split_function(data, split_ratio=0.8)
#Make prediction for desired point , for example: second point of dataest
X_ts_1 = x_ts.iloc[1]
x_ts_1_actual_label = y_ts.iloc[1]
x_ts_1_predicted_label = predict(x_tr,y_tr,X_ts_1,3)
print(x_ts_1_predicted_label)
```