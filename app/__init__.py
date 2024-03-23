from flask import Flask

app = Flask(__name__)

# Import views after creating the app object to avoid circular imports
from app.client import views