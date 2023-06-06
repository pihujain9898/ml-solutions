from flask import Flask, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Flask initialisation
app = Flask(__name__)
app.debug = True
app = Flask(__name__, template_folder='templates/templateFiles', static_folder='templates/staticFiles')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'fewdghr^#324dfg2342354)^}fswef324r'

# Application routes
@app.route("/")
def index():
    return redirect(url_for("upload"), code=301)
    # return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("upload"), code=301)

@app.route("/poperror")
def poperror():
    session.pop('error', None)
    return "Success"

@app.route("/preprocess")
def preprocess():
    if session.get('dataset'):
        if session.get('preprocess') or request.args.get('preprocess'):
            session['preprocess'] = session.get('preprocess') or request.args.get('preprocess')
            preprocess = session.get('preprocess')
            df = pd.read_csv(session.get('dataset'))
            if session.get('ignore_columns') or request.args.getlist('ignore_columns'):
                session['ignore_columns'] = session.get('ignore_columns') or request.args.getlist('ignore_columns')
                ignore_columns = session.get('ignore_columns') or request.args.getlist('ignore_columns')
                df = df.drop(ignore_columns, axis=1)

            model_list = ['Logistic Regression','Decision Tree','Random Forest','K-Neighbors','Linear SVM']             
            selected_model = request.args.get('selected_model')

            if selected_model is not None:
                # replace_data = json.loads(json.dumps(list(df[preprocess].value_counts().keys())))
                # df[preprocess] = df[preprocess].replace(replace_data)
                # return json.dumps(list(df[preprocess].value_counts().keys()))

                y = df[preprocess]
                X = df.drop(preprocess,axis=1)            
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.3, shuffle=True, random_state=43)            
                scaler = StandardScaler()
                scaler.fit(X_train)            
                X_train = pd.DataFrame(scaler.transform(X_train),columns = X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test),columns = X_test.columns, index=X_test.index)

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                if selected_model == model_list[0]:
                    cm, accuracy, precision, recall, score = logistic_regression(X_train, X_test, y_train, y_test)
                elif selected_model == model_list[1]:
                    cm, accuracy, precision, recall, score = decision_tree(X_train, X_test, y_train, y_test)
                elif selected_model == model_list[2]:
                    cm, accuracy, precision, recall, score = random_forest(X_train, X_test, y_train, y_test)
                elif selected_model == model_list[3]:
                    cm, accuracy, precision, recall, score = k_neighbours(X_train, X_test, y_train, y_test)
                elif selected_model == model_list[4]:
                    cm, accuracy, precision, recall, score = linear_svm(X_train, X_test, y_train, y_test)
                
                return render_template('result.html', cm=cm, accuracy=accuracy, precision=precision, recall=recall, score=score, selected_model=selected_model)
            else:
                return render_template('preprocess.html', selected_model=selected_model, model_list=model_list)
        else:
            session['error'] = "Select the preprocess column"
            return redirect("/eda", code=301)
    else:
        session['error'] = "First upload the dataset"
        return redirect("/upload", code=301)
    
def logistic_regression(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train.values.ravel())
    y_pred = log_reg.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy= accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    score = f1_score(y_test, y_pred, average='weighted')
    return cm, round(accuracy*100, 2), round(precision*100, 2), round(recall*100, 2), round(score*100, 2)

def decision_tree(X_train, X_test, y_train, y_test):
    classifier1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 5)
    classifier1.fit(X_train, y_train)
    y_pred = classifier1.predict(X_test)
    accuracy= accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    score = f1_score(y_test, y_pred, average='weighted')
    return cm, round(accuracy*100, 2), round(precision*100, 2), round(recall*100, 2), round(score*100, 2)

def random_forest(X_train, X_test, y_train, y_test):
    classifier_Random = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 5)
    classifier_Random.fit(X_train, y_train)
    y_pred = classifier_Random.predict(X_test)
    accuracy= accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    score = f1_score(y_test, y_pred, average='weighted')
    return cm, round(accuracy*100, 2), round(precision*100, 2), round(recall*100, 2), round(score*100, 2)

def k_neighbours(X_train, X_test, y_train, y_test):
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier =  KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy= accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    score = f1_score(y_test, y_pred, average='weighted')
    return cm, round(accuracy*100, 2), round(precision*100, 2), round(recall*100, 2), round(score*100, 2)

def linear_svm(X_train, X_test, y_train, y_test):
    classifier2 = SVC(kernel = 'rbf', random_state = 5, probability=True)
    classifier2.fit(X_train, y_train)
    y_pred = classifier2.predict(X_test)
    accuracy= accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    score = f1_score(y_test, y_pred, average='weighted')
    return cm, round(accuracy*100, 2), round(precision*100, 2), round(recall*100, 2), round(score*100, 2)

@app.route("/eda")
def eda():
    if session.get('dataset'):
        try:
          orignal_df = pd.read_csv(session.get('dataset'))
          df = pd.read_csv(session.get('dataset'))
        except:
          orignal_df = pd.read_excel(session.get('dataset'))
          df = pd.read_excel(session.get('dataset'))
          
        ignore_columns = request.args.getlist('ignore_columns')
        if len(ignore_columns)>0:
            session['ignore_columns'] = ignore_columns
            df = df.drop(ignore_columns, axis=1)
        preprocess = request.args.get('preprocess')
        if preprocess is not None:
            session['preprocess'] = preprocess

        # print(df.shape[1]-len(ignore_columns))
        return render_template('eda.html',  
                               head=[df.head().to_html(border=0, classes="table table-striped")],  
                               tail=[df.tail().to_html(border=0, classes="table table-striped")], 
                               df_titles=df.columns.values, 
                               describe=[df.describe().to_html(border=0, classes="table table-striped")],
                               # correlation=[df.corr().to_html(border=0, classes="table table-striped")],
                               titles=orignal_df.columns.values,
                               ignore_columns=session.get('ignore_columns') or [],
                               preprocess=session.get('preprocess'),
                               shape=df.shape
                               )
    else:
        session['error'] = "First upload the dataset"
        return redirect("/upload", code=301)

@app.route('/upload',methods = ['POST', 'GET'])
def upload():
  if request.method == 'GET':
     return render_template('upload.html')
  if request.method == 'POST':
    try:      
        f = request.files['dataset']
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        session['dataset'] = file_path
        # return file_path
        return redirect(url_for("eda"), code=301)
    except:
        session['error'] = "Somwthing went wrong while file upload"
        return redirect("/upload", code=301)
    #   return "Something went wrong", 200  
      
# App run
if __name__ == "__main__":
    app.run(host= '0.0.0.0')

