import os
from flask import (
     Flask, 
     request, 
     render_template)
from model import recommend #model.pyからrecommend関数をインポート

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) #トップページのルーティング
def top():
    return render_template('top.html')

@app.route('/kibun1', methods=['GET'])   #気分入力ページのルーティング
def kibun1():
    return render_template('kibun1.html')
    
@app.route('/kibun2', methods=['GET','POST'])   #出力ページのルーティング
def kibun2():
    if request.method == "GET":
        return render_template('kibun2.html')
    elif request.method == "POST":
        favs = request.form.getlist("fav")#name属性がfavのcheckboxから複数の値を取得
        data_str = ",".join(favs)
        name,item,introduction_text,image = recommend(data_str)
        return render_template('kibun2.html', name=name,item=item,introduction_text=introduction_text,image=image)#左辺がHTML、右辺がPython側の変数

if __name__ == "__main__":
    app.run(debug=True)