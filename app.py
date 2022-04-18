from distutils.log import error
from flask import Flask,render_template,request,url_for
import numpy as np
import pickle
import configModel as map_crop


app=Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))
map={}

@app.route('/')
def index():
    return render_template('index.html',show="none")

@app.route('/apple')
def apple():
    return render_template('apple.html')

@app.route('/banana')
def banana():
    return render_template('banana.html')

@app.route('/mango')
def mango():
    return render_template('mango.html')

@app.route('/index')
def index1():
    return render_template('index.html',show="none")

@app.route('/indexMobile')
def index_mobile():
    return render_template('indexMobile.html',show="none")

if __name__=="__main__":
    app.run(debug=True)

# def load_map():
#     global map
#     try:
#         map=pickle.load(open('map.pkl','rb'))
#     except(error):
#         print(error)

# def convert(key):
#     global map
#     res=0
#     try:
#         res=map[key]
#     except(error):
#         print(error)
#     return res

# def load_crops():
#     crops=[]
#     try:
#         crops=pickle.load(open('crops.pkl','rb'))
#         map_crop.set(crops)
#     except(error):
#         print(error)
#     return crops

@app.route('/predict',methods=['POST','GET'])
def predict():
    # load_map()
    # features=[convert(x) for x in request.form.values()]
    # features.append(0)
    # crops=load_crops()
    max_crop="Test"
    max_price=0
    # for i in crops:
    #     values=features.copy()
    #     values.append(convert(i))
    #     final=[np.array(values)]
        # pred=model.predict(final)
        # if(pred>max_price):
        #     max_price=pred
        #     max_crop=i

    text_crop="The predicted fruit or vegetable is"
    text_price="and the predicted price is"
    # max_crop,max_price=map_crop.get(max_crop,max_price)
    return render_template('index.html',labelCrop=text_crop,crop=max_crop,
        labelPrice=text_price,price=max_price)