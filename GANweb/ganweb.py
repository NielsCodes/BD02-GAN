from typing import Text
from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

#Homepage
@app.route("/")
def home():
    return render_template("index.html")

#Homepage with user input
@app.route("/", methods=["POST"])
def home_post():
    input = request.form.get("input")
    return render_template("index.html", forward_message=input)

#About us page
@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

#About GAN page
@app.route("/aboutgan")
def aboutgan():
    return render_template("aboutgan.html")

#Error 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

#Error 500
@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500    

#In production: get rid of debug=True
if __name__ == "__main__":
    app.run(debug=True)