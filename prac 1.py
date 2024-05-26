#importing the recq models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Cancer.csv')

#assigning x and y
y=df['diagnosis']
x=df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]

#training and predicting
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2529)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#finding the accuracy
print("accuracy score=",accuracy_score(y_test,y_pred))

#find our own solution
a = model.predict([[8.196,16.84,51.71,201.9,0.086,0.05943,0.01588,0.005917,0.1769,0.06503,0.1563,0.9567,1.094,8.205,0.008968,0.01646,0.01588,0.005917,0.02574,0.002582,8.964,21.96,57.26,242.2,0.1297,0.1357,0.0688,0.02564,0.3105,0.07409]])
if a=='M':
    print("Third Stage")
else:
    print("Second Stage")



#inorder to enter your own data please remove the values inside the square bracket and enter the data according to x

      

