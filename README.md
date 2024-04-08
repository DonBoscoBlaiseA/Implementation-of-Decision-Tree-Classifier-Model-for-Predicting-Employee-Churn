# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import libraries and use LabelEncoder to transform categorical 'salary' data into numerical labels.
2.  Choose relevant features ('satisfaction_level', 'last_evaluation', etc.) and target variable ('left') from the dataset.
3. Divide the dataset into training and testing sets using train_test_split with specified test size and random state.
4. Initialize a DecisionTreeClassifier with 'entropy' criterion and fit it to the training data.
5. Predict 'left' values for testing data, calculate accuracy using metrics.accuracy_score, and make predictions for new input features.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Don Bosco Blaise A
RegisterNumber: 212221040045
*/

import pandas as pd
data=pd.read_csv("G:/jupyter_notebook_files/employee_churn/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=data['left']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/DonBoscoBlaiseA/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/140850829/b3237a3a-f019-4bcc-a56c-295183255220)
![image](https://github.com/DonBoscoBlaiseA/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/140850829/a04454b8-ae4f-47f4-9c3b-917668497a73)
![image](https://github.com/DonBoscoBlaiseA/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/140850829/97d75d8a-7567-4990-b23c-887ff55bb3c6)
![image](https://github.com/DonBoscoBlaiseA/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/140850829/dd948df7-f7c1-4489-bd18-cab8e3e86bb3)
![image](https://github.com/DonBoscoBlaiseA/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/140850829/5b7859b4-5a15-4266-8eac-d82c56b3aa14)
![image](https://github.com/DonBoscoBlaiseA/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/140850829/408f3335-ea16-437c-a78c-6d508ccdcfa5)
![image](https://github.com/DonBoscoBlaiseA/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/140850829/f0d0c76b-0f7e-4577-a794-6184f599dfdb)
![image](https://github.com/DonBoscoBlaiseA/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/140850829/9f7b4f63-b390-4080-8f52-6e44de29ed38)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
