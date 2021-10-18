from flask import Flask, request
from flask_restful import Api, Resource, abort
from PIL import Image
import os
import uuid
import base64

from cgan_mnist import send_file

app = Flask(__name__)
api = Api(app)

def split_string(string): 
  return list(string)

def merge_images(images):
  image_count = len(images)
  image_dimension = 220
  height = image_dimension
  width = image_dimension * image_count

  output_img = Image.new(mode="RGB", size=(width, height))

  for idx, path in enumerate(images):
    img = Image.open(path)
    img.show()
    offset_x = idx * image_dimension
    output_img.paste(img, (offset_x, 0))

  output_path = 'output_image/' + str(uuid.uuid4()) + '.jpg'
  output_img.save(output_path)

  return output_path


def handle_request(value):
  # TODO: Add error handling

  # split input string into characters
  chars = split_string(value)

  # Array storing all individual digit generated image paths
  paths = []

  # convert character to int and use it to generate an image
  for char in chars:
    digit = int(char)

    image_path = send_file(digit)
    paths.append(image_path)

  # merge images
  merged_img_path = merge_images(paths)

  # remove individual digit images
  for path in paths:
    os.remove(path)

  # turn merged output image into Base64 string
  with open(merged_img_path, 'rb') as merged_img:
    encoded_img_bytes = base64.b64encode(merged_img.read())
    encoded_img_string = encoded_img_bytes.decode('utf-8')

  # remove output image file
  os.remove(merged_img_path)

  return encoded_img_string


class MnistHandler(Resource):
    def post(self):
        userInput = request.form["Value"]
        if len(userInput) > 10:
          abort(400, message='Request too long. Max 10 digits.')
        encoded_output_img = handle_request(userInput)
        return {"Value": encoded_output_img}

api.add_resource(MnistHandler, "/mnist")

if __name__ == "__main__":
    app.run(debug=True,port=8080, host="0.0.0.0")