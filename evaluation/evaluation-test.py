from mmdet.apis import init_detector, inference_detector
import os
import numpy as np
import requests
import base64
import time
import json
import sys
from utils import *
from constants import *

def get_file_extension(file_path):
   return os.path.splitext(file_path)[1]

def get_file_name_no_extension(file_path):
   return os.path.splitext(file_path)[0].split('/')[-1]

# Version 1
# Searches for all folders and files in /mmdetection/configs for a file that matches
# the name of the checkpoint file, ensuring that the adecuate configuration is used
# for the given NN.
# Works if the checkpoint file has the exact same name as the config file
def get_config_file_1(nn_checkpoint_name):
  # Splits the file name in name and extension
  nn_checkpoint_name = get_file_name_no_extension(nn_checkpoint_name)

  match_path = ""
  found_match = False
  # Iterates through all files in config folder
  for folder in os.listdir(CONFIGS_PATH):
      folder_path = f'{CONFIGS_PATH}/{folder}'
      # Continues if it is a folder that is not _user_
      if os.path.isdir(folder_path) and folder != '_user_':
        # Iterates through all the files in the folder
        for file_name in os.listdir(folder_path):
          # Splits the file name in name and extension
          file_name_short, file_name_ext = os.path.splitext(file_name)
          # Continues if it is a python file whose entire name is in the checkpoint's name
          if file_name_ext == ".py" and file_name_short == nn_checkpoint_name:
              match_path = f'{folder_path}/{file_name}'
              found_match = True
              break
        
        if found_match:
           break
  
  return match_path

# Version 2
# Searches for all folders and files in /mmdetection/configs for a file that matches
# the name of the checkpoint file, ensuring that the adecuate configuration is used
# for the given NN. It breaks down the name by the delimeter "_" and searches for matches up to
# a given part of the name.
# Works even when there is some identification data after the name of the NN in the checkpoint
# file name. Drawback is that it is not so precise, since it chooses the last match of the last 
# builded name. In some cases this may not be the correct configuration
def get_config_file_2(nn_checkpoint_name):
  # Splits the file name in name and extension
  nn_checkpoint_name = get_file_name_no_extension(nn_checkpoint_name)

  splitted_checkpoint_name = nn_checkpoint_name.split('_')

  last_match_path = ""

  for idx, name_part in enumerate(splitted_checkpoint_name):
    found_match = False

    builded_name = build_partial_name(splitted_checkpoint_name, idx)

    # Iterates through all files in config folder
    for folder in os.listdir(CONFIGS_PATH):
        folder_path = f'{CONFIGS_PATH}/{folder}'
        # Continues if it is a folder that is not _user_
        if os.path.isdir(folder_path) and folder != '_user_':
          # Iterates through all the files in the folder
          for file_name in os.listdir(folder_path):
            # Splits the file name in name and extension
            file_name_short, file_name_ext = os.path.splitext(file_name)
            # Continues if it is a python file whose entire name is in the checkpoint's name
            if file_name_ext == ".py" and builded_name in file_name_short:
                last_match_path = f'{folder_path}/{file_name}'
                found_match = True
                break
          
          if found_match:
             break
    
    if found_match:
       continue
    else:    
      break
  
  return(last_match_path)


def get_config_file(nn_checkpoint_name):
  config_file = get_config_file_1(nn_checkpoint_name)
  if not config_file:
    config_file = get_config_file_2(nn_checkpoint_name)

  return config_file


def build_partial_name(splitted_name, last_index):
  builded_name = ""
  for idy in range(last_index + 1):
    builded_name += splitted_name[idy]

    if idy != last_index:
      builded_name += "_"
  
  return builded_name


def run_inference(network_id, image_id, output_file_name, timer_file):  
  img_path = get_image_path(image_id)
  config_path = get_config_path(network_id)
  checkpoint_path = get_checkpoint_path(network_id)
  output_res_path = f"{OUTPUT_PATH}/{output_file_name}-res.json"
  output_ann_path = f"{OUTPUT_PATH}/{output_file_name}-ann.json"
  output_img_path = f"{OUTPUT_PATH}/{output_file_name}.png"

  # Build the model from a config file and a checkpoint file
  timer = time.time()
  model = init_detector(config_path, checkpoint_path)
  timer_file.write(f"init_detector(): {time.time() - timer} seconds\n")
  
  # Test a single image and show the results
  timer = time.time()
  results = inference_detector(model, img_path)
  timer_file.write(f"inference_detector(): {time.time() - timer} seconds\n")

  # Save Image Output
  model.show_result(img_path, results, out_file=output_img_path)

  # Save parsed results in json
  parsed_results = parse_results_to_dict(results)
  with open(output_res_path, 'w') as file:  
    json.dump(parsed_results, file, indent=2)
  
  # Save annotations in json
  annotations = convert_parsed_results_to_annotations(network_id, parsed_results)
  with open(output_ann_path, 'w') as file:  
    json.dump(annotations, file, indent=2)
        
  return


def main():
  if len(sys.argv) != 3:
    print("You must give 2 arguments: network_id and image_name")
    return
  
  network_id, image_id = sys.argv[1], sys.argv[2]

  output_file_name = f"{network_id}-{image_id}"
  print(f"Output File Name = {output_file_name}")

  timer_file = open(f"{OUTPUT_PATH}/{output_file_name}.txt", "a")

  timer = time.time()
  run_inference(network_id, image_id, output_file_name, timer_file)
  timer_file.write(f"run_inference(): {time.time() - timer} seconds\n")

  print("FINISHED SUCCESSFULLY")


if __name__ == "__main__":
  main()