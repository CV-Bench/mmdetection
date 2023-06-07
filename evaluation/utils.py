import base64
import uuid
import os
import json
import numpy as np
from constants import *


def get_network_type_from_id(network_id):
  return network_id.split('-')[0]


def parse_evaluation_results(network_id, results):
  parsed_results = parse_results_to_dict(results)
  annotations = convert_parsed_results_to_annotations(network_id, parsed_results)
  return annotations


def convert_parsed_results_to_annotations(network_id, results):
  categories_file = open(get_categories_path(network_id), "r")
  categories = json.load(categories_file)

  annotations = []
  for category_index, boxes_values in results.items():
    for box_values in boxes_values:
      annotations.append({
      "coordinates": {
          "min_x": box_values[0],
          "min_y": box_values[1],
          "max_x": box_values[2],
          "max_y": box_values[3]
      },
      "label": categories[str(category_index)],
      "confidence": box_values[4]
      })
          
  return annotations


def encode_image(image_path):
  image_file = open(image_path, "rb")
  encoded_string = base64.b64encode(image_file.read())
  return encoded_string


def delete_file(file_path):
  os.remove(file_path)


def save_image(image_encoded):
  image_id = uuid.uuid4()
  image_path = f"{IMAGES_PATH}/{image_id}.png"

  with open(image_path, "wb") as f:
    image = base64.b64decode(image_encoded)
    f.write(image)

  return image_path


def get_categories_path(network_id):
  return f"{CATEGORIES_PATH}/{network_id}.json"


def get_checkpoint_path(network_id):
  return f"{CHECKPOINTS_PATH}/{network_id}.pth"


def get_config_path(network_id):
  network_type = get_network_type_from_id(network_id)
  return f"{CONFIGS_PATH}/{network_type}.py"


def get_image_path(image_id):
  return f"{IMAGES_PATH}/{image_id}.png"


def parse_results_to_dict(results):
  parsed_results = {}

  category_index = 0
  for item in results:
    if type(item) == np.ndarray:
      if len(item) > 0:
        if type(item[0]) == np.ndarray:
          parsed_results[category_index] = []
          for i in item:
            parsed_results[category_index].append(i.tolist())
        else:
          parsed_results[category_index] = item.tolist()
        category_index += 1
      else:
        parsed_results[category_index] = []
        category_index += 1
    else:
      for i in item:
        parsed_results[category_index] = i.tolist()
        category_index += 1
      break

  return parsed_results