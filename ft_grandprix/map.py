import argparse
import json
import em
from .colors import colors
import random

def r():
    return random.choices(list(range(256)), k=3)

def produce_mjcf(
          template_path = "models/car.em.xml",
          metadata_path = "models/chunks/metadata.json",
          mjcf_metadata_path = "models/car.json",
          mjcf_path = "models/car.xml",
          rangefinders = 90,
          cars = None
):
    if cars is None:
        cars = [
                { "x" : 5.6 ,"y" : 0.1, "primary" : colors["red"],    "secondary" : colors["pink"]       },
                { "x" : 5.5, "y" : 0.0, "primary" : colors["orange"], "secondary" : colors["darkorange"] },
                # { "x" : 5.5, "y" : 0.1, "primary" : colors["blue"],   "secondary" : colors["green"]      },
                # { "x" : 5.6 ,"y" : 0.1, "primary" : colors["purple"], "secondary" : colors["yellow"]     },
                # { "x" : 5.5, "y" : 0.0, "primary" : r(), "secondary" : r() },
                # { "x" : 5.5, "y" : 0.1, "primary" : r(),   "secondary" : r()      },
                # { "x" : 5.6 ,"y" : 0.1, "primary" : r(),    "secondary" : r()       },
                # { "x" : 5.6 ,"y" : 0.1, "primary" : r(), "secondary" : r()     }
               ]
    with open(metadata_path) as metadata_file:
         metadata = json.load(metadata_file)
    
    with open(mjcf_path, "w") as mjcf_file:
         with open(template_path) as template_file:
             interpreter = em.Interpreter(output=mjcf_file)
             interpreter.file(
                 template_file,
                 locals={
                     "cars" : cars,
                     "metadata" : metadata,
                     "rangefinders" : rangefinders,
                 })
             interpreter.shutdown()
    
    with open(mjcf_metadata_path, "w") as mjcf_metadata_file:
         mjcf_metadata = {
              "cars" : cars,
              "rangefinders" : rangefinders,
         }
         json.dump(mjcf_metadata, mjcf_metadata_file)

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     args = parser.parse_args()
     produce_mjcf()
