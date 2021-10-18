from flask import Flask, request                # Flask for API framework
from flask_restful import Api, Resource, abort  # Flask for API framework
from PIL import Image                          # PIL (Pillow in requirements.txt) for image manipulation (merging multiple images into one)
import os                                      # Used for file system interaction
import uuid                                    # Used to generate Universial Unique IDs for generated image to prevent multiple requests using the same filename
import base64                                  # Used to encode generated image as Base64 string to send to frontend client
import re                                      # Used for Regular Expression matching of request strings

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


def handle_mnist_gen_request(value):
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
    """Request handler class to handle all MNIST dataset related requests"""
    def post(self):
        """Request handler for MNIST generation POST requests

          - verifies requested string does not exceed max length (10 digits)
          - verifies requested string does not contain non-digits using Regular Expression 
        """
        user_input = request.form["value"]

        if len(user_input) > 10:
          abort(400, message='Request too long. Max 10 digits.')
        
        non_digit_matches = re.search(r"\D", user_input)
        if non_digit_matches is not None:
          abort(400, message='Request invalid. Should only contain digits (0-9).')

        encoded_output_img = handle_mnist_gen_request(user_input)
        return {
          'message': 'Image generated successfully',
          'img': encoded_output_img,
          }

api.add_resource(MnistHandler, "/mnist")

if __name__ == "__main__":
    app.run(debug=True,port=8080, host="0.0.0.0")