from app import app
from flask import render_template

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/loading_page', methods=['GET'])
def loading_page():
    return render_template('loading.html')

@app.route('/list', methods=['GET'] )
def list():
    return render_template('list.html')

@app.route('/videos', methods=['GET'] )
def videos():
    return render_template('videos.html')