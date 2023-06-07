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


    def on_evaluate(self, request_data):
        encoded_image = request_data["image"]
        network_id = request_data["network_id"]
        # TODO: Implement caching system so that only one model can be loaded in cache per user. Use user_id for that
        user_id = request_data["user_id"]
        evaluation_id = request_data["evaluation_id"]
        
        if self.network_is_loaded(network_id):
            self.loaded_networks[network_id]["last_used"] = datetime.now()
        else:
            self.load_network(network_id)

        results = self.run_evaluation(network_id, encoded_image)
        
        # self.send_results(evaluation_id, results)

        return {"evaluation_id": evaluation_id, "results": results}


    def network_is_loaded(self, network_id):
      return network_id in self.loaded_networks.keys()
    
    
    def load_network(self, network_id):
      checkpoint_path = get_checkpoint_path(network_id)
      config_path = get_config_path(network_id)

      self.loaded_networks[network_id] = {
          "model": init_detector(config_path, checkpoint_path),
          "last_used": datetime.now()
      }


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


    def run_evaluation(self, network_id, image_encoded):
      model = self.loaded_networks[network_id]["model"]

      image_path = save_image(image_encoded)
      
      results = inference_detector(model, image_path)

      bb_image_path = f"/data/images/{uuid.uuid4()}.png"
      model.show_result(image_path, results, out_file=bb_image_path)
      
      encoded_bb_img = encode_image(bb_image_path)

      delete_file(bb_image_path)
      delete_file(image_path)

      annotations = parse_evaluation_results(network_id, results)

      return {"image": encoded_bb_img, "annotations": annotations}