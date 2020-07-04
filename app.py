from flask import Flask, url_for, render_template, redirect, session, Response, jsonify #将内容转换为json
from flask.views import request
import os, json
from Word2vec import AutoSummary

app = Flask(__name__)

BASE_DIR = './data'

@app.route('/index')
def index():
    """
    首页
    :return:
    """
    if session.get('is_login', None):
        return render_template('index.html')
    return redirect(url_for('login'))


@app.route('/login', methods=['GET','POST'])
def login():
    """
    登录
    :return:
    """
    print('path', request.path)
    print('headers', request.headers)
    print('method', request.method)
    print('url', request.url)
    print('data', request.form)

    if request.method == "POST":
        # username = request.form['username']
        # password = request.form['password']
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        print(username, password)
        if username == "root" and password == "123":
            session['username'] = username
            session['password'] = password
            session['is_login'] = True
            return redirect(url_for('index'))
        else:
            return "Invalid username/password"

    return render_template('login.html')

@app.route('/logout')
def logout():
    """
    注销
    :return:
    """
    session.pop('username',None)
    print('logout')
    return redirect(url_for('login'))

@app.route('/submit_news', methods=['GET','POST'])
def get_news_from_input():
    """
    输入方式获取
    :return:
    """
    status = False
    message = "Response Good"
    if request.method == "POST":
        res = dict()
        sub_title = request.form['title']
        sub_news = request.form['news']
        try:
            summary = AutoSummary()
            result = summary.get_summary(sub_news, sub_title, constraint=200, algorithm='Cosine', use_sif=True)
            res['summary'] = result
            status = True
            message = "error message"
        except Exception as e:
            print(e)
        res['status'] = status
        res['message'] = message
        return jsonify(res)

# set the secret key.  keep this really secret:
app.secret_key = os.urandom(24)


if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)
