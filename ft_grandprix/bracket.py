import os
import json
from .colors import colors as _colors
_colors = [a[1] for a in sorted(list(_colors.items()), key=lambda t: t[0])]
# breakpoint()
# exit()

seed =  10
items = []
files = sorted(os.listdir("drivers"))
# print(files)

def h(s):
    global seed
    return 1 if s == "" else (((seed + 101 * ord(s[0])) % 2003) * h(s[1:])) % 1009

for index, file in enumerate(files):
    if ".py" not in file or "__" in file:
        continue
    stripped = file[:-3]
    module = f"drivers.{stripped}"
    item = dict(
        driver = module,
        name = module,
        primary = _colors[h(module) % len(_colors)],
        secondary = _colors[h(module + ".") % len(_colors)],
        icon = "white.png"
    )
    with open(f"drivers/{stripped}.json", "w") as f:
        json.dump(item, f)
    items.append(item)

final =  sorted(items, key=lambda k: h(k["name"]))
# print(h("a"))
# print(files)
print (json.dumps(final))
