![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/9acbde61-8bb6-4de8-a5d7-3821ffdc3bef)![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/545715be-8e6c-4cb4-a1ef-afde13c6a832)![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/c05254bf-ef6e-45df-9e4d-7c4c75f4cebd)# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Find the null and duplicate values.
3. Using logistic regression find the predicted values of accuracy , confusion matrices.
4. Display the results.

 
## PROGRAM:
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:T DANUSH REDDY 
RegisterNumber:212223040029  
*/
...
## OUTPUT:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/e81daab9-37a3-4cc9-bdaf-60937434d7ee)


![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/04fc41e8-e74a-439e-9975-83f3a525c578)



## RESULT:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
