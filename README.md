# parkinsons_project
#import libraries
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('parkinsons.csv')
X=dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,12,14,15,16,18,19,20,21,22,23]].values
y=dataset.iloc[:,17].values

#splitting into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state =0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting into KNN model
from sklearn.neighbors import KNeighborsClassifier
classifi = KNeighborsClassifier(n_neighbors = 8,p=2,metric ='minkowski')
classifi.fit(X_train,y_train)

#predicting reults
y_pred = classifi.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#f1 score and accuracy score
from sklearn.metrics import f1_score,accuracy_score
f1_score(y_test,y_pred)
accuracy_score(y_test,y_pred)


#Fitting ther data in naive bayes
from sklearn.naive_bayes import GaussianNB
classifi1 = GaussianNB()
classifi1.fit(X_train,y_train)

#predicting reults
y1_pred = classifi1.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y1_pred)

#f1 score and accuracy score
from sklearn.metrics import f1_score,accuracy_score
f1_score(y_test,y1_pred)

accuracy_score(y_test,y1_pred)


#fitting the model in SVM
from sklearn.svm import SVC
classifi2 = SVC()
classifi2.fit(X_train,y_train)

#predicting reults
y2_pred = classifi2.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(y_test,y2_pred)

#f1 score and accuracy score
from sklearn.metrics import f1_score,accuracy_score
f1_score(y_test,y2_pred)

accuracy_score(y_test,y2_pred)

#fitting the data in random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifi3 = RandomForestClassifier(n_estimators=16,criterion = "entropy",random_state=0)
classifi3.fit(X_train,y_train)

#predicting reults
y3_pred = classifi3.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm3=confusion_matrix(y_test,y3_pred)

#f1 score and accuracy score
from sklearn.metrics import f1_score,accuracy_score
f1_score(y_test,y3_pred)

accuracy_score(y_test,y3_pred)


## Pickle
import pickle

# save model
pickle.dump(classifi, open('Parkinsons_Disease_Detector.pickle', 'wb'))

# load model
Parkinsons_Disease_Detector_model = pickle.load(open('Parkinsons_Disease_Detector.pickle', 'rb'))

# predict the output
yf_pred = Parkinsons_Disease_Detector_model.predict(X_test)

# confusion matrix
print('Confusion matrix of KNN Classifier: \n',confusion_matrix(y_test, yf_pred),'\n')

# show the accuracy
print('Accuracy of KNN Classifier = ',accuracy_score(y_test, yf_pred))
