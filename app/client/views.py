from app import app
from flask import render_template

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/loading')
def loading_page():
    return render_template('loading.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/review')
def review_page():
    return render_template('review.html')