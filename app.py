from flask import Flask,render_template,request,url_for
import numpy as np


app=Flask(__name__)

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

if __name__=="__main__":
    app.run(debug=True)