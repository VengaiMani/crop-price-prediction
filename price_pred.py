import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_set=pd.read_csv("agree.csv")
column_means = data_set.mean()
data_set= data_set.fillna(column_means)
data_set.drop('production', inplace=True, axis=1)
data_set.drop('area', inplace=True, axis=1)

le=LabelEncoder()
data_set['state_name']=le.fit_transform(data_set['state_name'])
data_set['district_name']=le.fit_transform(data_set['district_name'])
data_set['crop_year']=le.fit_transform(data_set['crop_year'])
data_set['season']=le.fit_transform(data_set['season'])
data_set['crop']=le.fit_transform(data_set['crop'])
x=data_set.iloc[:,[0,5]].values
y=data_set.iloc[:,[5]].values
y=y.astype('int')  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=10)  

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 
from sklearn.svm import SVC   
classifier = SVC(kernel='linear')  
classifier.fit(x_train, y_train)

# from sklearn import metrics
# predictions=metrics.accuracy_score(y_test,y_pred)
# predictions