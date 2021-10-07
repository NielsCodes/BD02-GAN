from typing import Text
from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500    

if __name__ == "__main__":
    app.run(debug=True)