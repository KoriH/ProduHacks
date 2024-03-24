from flask import Flask
import os

template_dir = os.path.abspath('app/client/templates')
static_dir = os.path.abspath('app/client/static')
app = Flask(__name__, template_folder=template_dir,static_folder=static_dir)

from app.client import views