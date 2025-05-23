import os
import json
from typing import Tuple

class Hasher:
    def __init__(self, seed):
        self.seed = seed

    def hash(self, string):
        return 1 if string == "" else (((self.seed + 101 * ord(string[0])) % 2003) * self.hash(string[1:])) % 1009

def compute_driver_files(drivers_path, silent=False):
    from .colors import colors as _colors
    _colors = [a[1] for a in sorted(list(_colors.items()), key=lambda t: t[0])]

    hasher = Hasher(10)
    items = []
    files = sorted(os.listdir(drivers_path))    
    
    for file in files:
        if ".py" not in file or "__" in file:
            continue
        stripped = file[:-3]
        if not silent:
            print(f"Found candidate driver '{stripped}'")
        module = f"drivers.{stripped}"
        primary = _colors[hasher.hash(module) % len(_colors)]
        secondary = _colors[hasher.hash(module + ".") % len(_colors)]
        item = dict(
            driver = module,
            name = module,
            primary = primary,
            secondary = secondary,
            icon = "white.png"
        )
        output_path = f"drivers/{stripped}.json"
        if not silent:
            print(f"- Writing driver config to '{output_path}'")
        with open(output_path, "w") as f:
            json.dump(item, f)
        items.append(item)
    
    if not silent:
        print("Collection of all items")
        for item in items:
            print(f"- {item}")

if __name__ == "__main__":
    compute_driver_files("drivers")
