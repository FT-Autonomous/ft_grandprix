import argparse
import json
import em
from .colors import colors
import random

def produce_mjcf(
          template_path = "models/car.em.xml",
          metadata_path = "models/chunks/metadata.json",
          mjcf_metadata_path = "models/car.json",
          mjcf_path = "models/car.xml",
          rangefinders = 100,
          cars = None
):
    if cars is None:
        r = lambda: random.choices(list(range(256)), k=3)
        cars = [
            # { "x" : 5.5, "y" : 0.1, "primary" : colors["white"],    "secondary" : colors["lightblue"]  , "icon" : "trinity.png" },
            # { "x" : 5.6 ,"y" : 0.1, "primary" : colors["brown"],    "secondary" : colors["maroon"]     , "icon" : "nuim.png"    },
            # { "x" : 5.6 ,"y" : 0.1, "primary" : [2, 109, 153],      "secondary" : colors["blue"]       , "icon" : "tu.png"      },
            { "x" : 5.6 ,"y" : 0.1, "primary" : colors["red"],    "secondary" : colors["pink"]       , "icon" : "noob.png"    },
            { "x" : 5.5, "y" : 0.0, "primary" : colors["orange"], "secondary" : colors["darkorange"] , "icon" : "noob.png"    },
            { "x" : 5.5, "y" : 0.1, "primary" : colors["blue"],   "secondary" : colors["green"]      , "icon" : "noob.png"    },
            # { "x" : 5.6 ,"y" : 0.1, "primary" : colors["purple"], "secondary" : colors["yellow"]     , "icon" : "noob.png"    },
            # { "x" : 5.5, "y" : 0.0, "primary" : r(),              "secondary" : r()                  , "icon" : "noob.png"    },
            # { "x" : 5.5, "y" : 0.1, "primary" : r(),              "secondary" : r()                  , "icon" : "noob.png"    },
            # { "x" : 5.6 ,"y" : 0.1, "primary" : r(),              "secondary" : r()                  , "icon" : "noob.png"    },
            # { "x" : 5.6 ,"y" : 0.1, "primary" : r(),              "secondary" : r()                  , "icon" : "noob.png"    },
            # { "x" : 5.5, "y" : 0.0, "primary" : r(),              "secondary" : r()                  , "icon" : "noob.png"    },
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
