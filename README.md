# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Find the null and duplicate values.
3.Using logistic regression find the predicted values of accuracy , confusion matrices.
4.Display the results.

 
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
```

## Output:
1.Placement Data:
![238188357-d343df92-e10c-416a-bbeb-469c70f9317b](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/1c08057c-a09f-4f2d-9d66-982082096bfc)

2.Salary Data:
 ![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/aaaff8e4-728e-40c1-9bba-e114a73a6a40)

3.Checking the null function():
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/4124a7df-af62-4a61-bd6e-ebd48c3eaea5)
4.Data Duplicate:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/38d79d67-ca20-42a6-a1b8-2e7eac37cc5f)
5.PRINT DATA:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/0f05f30e-40b8-4cae-99b0-84b664921a7b)
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/0f657e5b-9f4f-42d1-91dc-6fe1f8d7d804)
6.Data Status:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/b43e8e49-d0a1-47fb-bd90-c30c43f8e3b5)
7.y_prediction array:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/4decedf3-c721-4088-886e-ec4087047b3b)
8.Accuracy value
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/cac92fb7-92c8-4546-ada1-cda580cd0ecb)
9.Confusion matrix:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/5d377d47-925f-4c1d-96ed-a6c225a0edcf)
10.Classification Report:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/543b4603-925a-49ff-b72e-cca33933e902)
11.Prediction of LR
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/f9567e2c-8b4f-4ced-950b-da8b104765a0)

## RESULT:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
