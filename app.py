from flask import Flask,redirect,url_for,render_template,request
from CFD_score import *
from CFD_Scoring_1 import *
from CNN_std import *

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template("index.html")



@app.route('/CNN',methods=['GET','POST'])
def CNN():
    if request.method == "POST":
        return render_template("CNN_Std2.html", CNN_Score=scor)
    return render_template("CNN_Std.html")



if __name__ == '__main__':
    app.run(debug=True)
