from flask import Flask, request
from flask_restful import Api, Resource
import requests

app = Flask(__name__)
api = Api(app)

BASE = "http://127.0.0.1:5000/"


class InputPut(Resource):
    def post(self):
        print(request.form["Value"])
        userInput = request.form["Value"]
        return {"Value": userInput}


api.add_resource(InputPut, "/mnistpost")


if __name__ == "__main__":
    app.run(debug=True,port=8000)