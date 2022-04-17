import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

data_set=pd.read_csv("agree.csv")
column_means = data_set.mean()
data_set= data_set.fillna(column_means)
data_set.drop('production', inplace=True, axis=1)
data_set.drop('area', inplace=True, axis=1)
le=LabelEncoder()
map={}
print("Starting to encode dataset")
le.fit(data_set['state_name'])
map.update(dict(zip(le.classes_,le.transform(le.classes_))))
data_set['state_name']=le.transform(data_set['state_name'])

le.fit(data_set['district_name'])
map.update(dict(zip(le.classes_,le.transform(le.classes_))))
data_set['district_name']=le.transform(data_set['district_name'])

le.fit(data_set['crop_year'])
map.update(dict(zip(le.classes_,le.transform(le.classes_))))
data_set['crop_year']=le.transform(data_set['crop_year'])

le.fit(data_set['season'])
map.update(dict(zip(le.classes_,le.transform(le.classes_))))
data_set['season']=le.transform(data_set['season'])

le.fit(data_set['crop'])
map.update(dict(zip(le.classes_,le.transform(le.classes_))))
crops=le.classes_
data_set['crop']=le.transform(data_set['crop'])

print("Starting to split train and test data")
x=data_set.iloc[:,[0,5]].values
y=data_set.iloc[:,[5]].values
y=y.astype('int')  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=10)  

print("Model creation started")
#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
print("Started to fit data")
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 
from sklearn.svm import SVC   
classifier = SVC(kernel='linear')  
classifier.fit(x_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    print("TRAINIG RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
clf3 = SVC(kernel='linear', probability=True)
bagging_clf = BaggingClassifier(base_estimator=clf3, n_estimators=1500, random_state=42)
bagging_clf.fit(x_train, y_train)
evaluate(bagging_clf, x_train, x_test, y_train, y_test)

from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(x_train, y_train, verbose=False)

predictions = my_model.predict(x_test)
from sklearn.metrics import mean_absolute_error
print("Mean absolute error : " + str(mean_absolute_error(predictions, y_test)))
print("Root mean squared error : "+str(np.sqrt(mean_absolute_error(predictions, y_test))))

# state_name=map["Andaman and Nicobar Islands"]
# district_name=map["NICOBARS"]
# crop_year=map[2000]
# season=map["Kharif"]
# crop=map["Arecanut"]

# pred=classifier.predict([[state_name,district_name,crop_year,season,crop]])
# print("Model created in price_pred")
# print(pred)

# from sklearn import metrics
# y_pred=classifier.predict(x_test)
# predictions=metrics.accuracy_score(y_test,y_pred)
# print(predictions)
# pickle.dump(classifier,open('model.pkl','wb'))

# with open('model.pkl','wb') as handle:
#     pickle.dump(classifier,handle,protocol=pickle.HIGHEST_PROTOCOL)
# with open('map.pkl','wb') as handle:
#     pickle.dump(map, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('crops.pkl','wb') as handle:
#     pickle.dump(crops,handle)