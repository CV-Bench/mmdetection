import socketio
import os
from datetime import datetime, timedelta
import base64
import uuid
import numpy as np
import json
from mmdet.apis import init_detector, inference_detector
from evaluation.utils import *


#TODO: Clear the cache of unused networks 3 minutes after it was last used
class EvaluationNamespace(socketio.AsyncClientNamespace):
    loaded_networks = {}

    def on_connect(self):
        print("Namespace /evaluation connected.")
        # logger.info("Namespace /evaluation connected.")


    def on_disconnect(self):
        print("Namespace /evaluation disconnected.")
        # logger.warning("Namespace /evaluation disconnected.")

    # Runs evaluation
    # Params:
    # - request_data:   Dictionary with the following values 
    # ["image", "network_id", "user_id", "evaluation_id", "preview"]
    # "image" is a base64 encoded image
    # "preview" is a boolean that decides if the bounding boxes image should be sent back to web app.

    def on_evaluate(self, request_data):
      try:
        network_id = request_data["network_id"]
        # TODO: Implement caching system so that only one model can be loaded in cache per user. Use user_id for that
        user_id = request_data["user_id"]
        evaluation_id = request_data["evaluation_id"]

        if self.network_is_loaded(network_id):
            self.loaded_networks[network_id]["last_used"] = datetime.now()
        else:
            self.load_network_to_cache(network_id)

        results = self.run_evaluation(request_data)
        # self.send_results(evaluation_id, results)
        return {"evaluation_id": evaluation_id, "results": results, "error": False, "error_msg": ""}
      
      except Exception as e:
          return {"evaluation_id": evaluation_id, "results": {}, "error": True, "error_msg": e}
        
        


    def network_is_loaded(self, network_id):
      return network_id in self.loaded_networks.keys()
    
    
    def load_network_to_cache(self, network_id):
      try:
        checkpoint_path = get_checkpoint_path(network_id)
        config_path = get_config_path(network_id)
        
        self.loaded_networks[network_id] = {
            "model": load_model(config_path, checkpoint_path),
            "last_used": datetime.now()
        }

      except Exception as e:
         raise Exception("Could not load the network to the cache.") from e
      
      
    def clear_unused_from_cache(self):
       now = datetime.now()
       for network_id in self.loaded_networks.keys():
          last_used = self.loaded_networks[network_id]["last_used"]

          if last_used + timedelta(hours=0, minutes=3) < now:
            self.loaded_networks.pop(network_id)


    def send_results(self, evaluation_id, results):
      self.emit(
        'evaluation_finished', 
        {
            "evaluation_id": evaluation_id,
            "results": results
        }
      )

    def run_evaluation(self, request_data):
      try:
        network_id = request_data["network_id"]
        encoded_image = request_data["image"]
        preview = request_data["preview"]

        model = self.loaded_networks[network_id]["model"]

        image_path = save_image(encoded_image)
        
        results = run_inference(model, image_path)

        encoded_bb_img = ""

        if preview:
          encoded_bb_img = get_preview_image(model, image_path, results)

        delete_file(image_path)

        annotations = parse_evaluation_results(network_id, results)

        return {"image": encoded_bb_img, "annotations": annotations}
      
      except Exception as e:
        raise Exception("Could not run evaluation correctly.") from e  
    
