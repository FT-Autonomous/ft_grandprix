import re
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import svg.path

np.set_printoptions(
    precision=2,
    formatter={"float": lambda x: f"{x:8.2f}"}
)

def extract_path_from_svg(path="template/what.xml", verbose=False):
    root = ET.parse(path).getroot()
    path = root[0].findall("path")
    namespace = re.match(r"\{(.+)\}", root.tag)
    if namespace:
        namespace = "{" + namespace.group(1) + "}"
    dee = root.find(f"{namespace}g").find(f"{namespace}path").attrib['d']

    pp = svg.path.parse_path(dee)
    ml = [pp.point(i/100) for i in range(100)]
    ml = np.array([[c.real, c.imag] for c in ml])
    return ml

def _extract_path_from_svg(path="template/what.xml", verbose=False):
    root = ET.parse(path).getroot()
    path = root[0].findall("path")
    namespace = re.match(r"\{(.+)\}", root.tag)
    if namespace:
        namespace = "{" + namespace.group(1) + "}"
    dee = root.find(f"{namespace}g").find(f"{namespace}path").attrib['d']

    pp = svg.path.parse_path(dee)
    ml = [pp.point(i/100) for i in range(100)]
    ml = np.array([[c.imag, c.real] for c in ml])
    plt.autoscale(enable=True, axis="y")
    plt.plot(ml[:, 1], -ml[:, 0])
    plt.show()
    breakpoint()
    # exit()
    
    path = dee.split()
    assert path.pop(0) == "m", "path should start with a move"
    def coord(s): return [float(x) for x in s.split(",")]
    x, y = sx, sy = coord(path.pop(0))
    arr = [[x, y]]
    assert path.pop(0) == "c", "path should be followed by cubic bezier control points"
    while True:
        n = path.pop(0)
        if "," not in n:
            break
        dx, dy = coord(n)
        # x += dx
        # y += dy

        x = x + dx
        y = y + dy
        
        if verbose:
            print(f"{x:9.3f},{y:9.3f} from {dx:9.3f},{dy:9.3f}")
        arr.append([x,y])
    arr.append([sx, sy])
    z = np.array(arr)
    breakpoint()
    
    interpolated = []
    for index in range(len(z)-1):
        a, b = z[index], z[index+1]
        interpolated.append(a)
        distance = np.linalg.norm(a - b)
        if distance > 50:
            delta = b -  a
            points = int(distance / 50)
            peakfiction = np.linspace(0, 1, 1 + points)
            for m in (peakfiction * delta[..., np.newaxis]).T[1:-1]: interpolated.append(a + m)
    interpolated.append(b)
            
    l = np.array(interpolated)
    # print(len(interpolated))
    # print(len(z))
    # plt.axis("fixed")
    plt.autoscale(enable=True, axis="y")
    plt.plot(z[:, 0], -z[:, 1])
    plt.show()
    
    # okay, let's say, arbitrarilly, that we want a marker every 50
    # units. So between any two points, if the distance is greater than 50
    # (x/50) > 0, lerp with values 0, …, 1 the step is SIZE/50

    # assertion is not really needed
    # assert n == "z", "track should close up at the end"

    return l

# print("Result: ", extract_path_from_svg())

if __name__ == "__main__":
    path = extract_path_from_svg(verbose=True)
    print(path)
