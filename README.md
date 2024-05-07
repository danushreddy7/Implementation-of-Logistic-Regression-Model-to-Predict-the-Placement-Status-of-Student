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
```

## Output:
# 1.Placement Data:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/85d984f4-15a1-4117-882e-35bcc0c2cb6a)
# 2.Salary Data:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/12b032aa-71e4-495c-a8e1-451d13a41fb8)
# 3.Checking the null function():
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/d2ea4bae-7390-40a2-bdab-d1c9fbd08961)
# 4.Data Duplicate:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/2c0985ed-5d02-40bf-a7df-d8dbcb418e1f)
# 5.Print Data:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/a6ec1285-d836-4f6d-a8e9-064e8a56b892)
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/91555ec7-efde-4fc2-8195-8e5d8f347438)
# 6.Data Status:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/94f9501e-58b6-4a5e-9231-e939b40c2045)
# 7.y_prediction array:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/ae3eee45-a7ec-4d0b-bfeb-df45f5adb9ed)
# 8.Accuracy value:
![image](https://github.com/danushreddy7/Implementation-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/6c64c1ad-099c-4544-b935-87cc7c786623)
# 9.Confusion matrix:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/1fe56b4f-46f7-4445-aa32-9bcd94cf9798)
# 10.Classification Report:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/1df66e31-4936-4cbe-a2ce-2818f04ef936)
# 11.Prediction of LR:
![image](https://github.com/danushreddy7/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149035740/ff3a164c-b025-4332-9b68-29fe4bf79fca)




## RESULT:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
