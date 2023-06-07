import base64
import uuid
import os
import json
import numpy as np
from constants import *
from mmdet.apis import init_detector, inference_detector


#TODO: Notify Jorin that it is also necessary to send to the Evaluation Docker the type of the network. Include it in the network ID separated by dash.
def get_network_type_from_id(network_id):
  try:
    return network_id.split('-')[0]
  except Exception as e:
    raise Exception("Could not get the network type from the network id.") from e


def parse_evaluation_results(network_id, results):
  try:
    parsed_results = parse_results_to_dict(results)
    annotations = convert_parsed_results_to_annotations(network_id, parsed_results)
    return annotations
  except Exception as e:
    raise Exception("Could not parse the evaluation results correctly.") from e


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
  try:
    image_id = uuid.uuid4()
    image_path = f"{IMAGES_PATH}/{image_id}.png"

    with open(image_path, "wb") as f:
      image = base64.b64decode(image_encoded)
      f.write(image)

    return image_path

  except Exception as e:
    raise Exception("Could not save encoded image in file system.") from e


def get_categories_path(network_id):
  return f"{CATEGORIES_PATH}/{network_id}.json"


def get_checkpoint_path(network_id):
  try:
    return f"{CHECKPOINTS_PATH}/{network_id}.pth"
  except Exception as e:
    raise Exception("Could not get checkpoint path.") from e


def get_config_path(network_id):
  try:
    network_type = get_network_type_from_id(network_id)
    return f"{CONFIGS_PATH}/{network_type}.py"
  except Exception as e:
    raise Exception("Could not get config path.") from e


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

def load_model(config_path, checkpoint_path):
  try:
    return init_detector(config_path, checkpoint_path)
  except Exception as e:
      raise Exception("Could not load the network model.") from e
  
def run_inference(model, image_path):
  try: 
    return inference_detector(model, image_path)
  except Exception as e:
    raise Exception("Could not run inference of the network model on the image.") from e
  

def get_preview_image(model, image_path, results):
  try:
    bb_image_path = f"/data/images/{uuid.uuid4()}.png"
    model.show_result(image_path, results, out_file=bb_image_path)
    encoded_bb_img = encode_image(bb_image_path)
    delete_file(bb_image_path)

    return encoded_bb_img
  except Exception as e:
      raise Exception("Could not get preview image withthe bounding boxes") from e