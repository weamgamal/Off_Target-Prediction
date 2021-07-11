from flask import Flask,redirect,url_for,render_template,request
from CFD_score import *
from CFD_Scoring_1 import *

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def Home():
    if request.method == "POST":
        return render_template("cfd2.html", CFD_Score=s)
    return render_template("CFD.html")





if __name__ == '__main__':
    app.run(debug=True)
