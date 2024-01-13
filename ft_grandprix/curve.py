import re
import numpy as np
import xml.etree.ElementTree as ET
import svg.path

def extract_path_from_svg(
        path="template/what.xml",
        points=100,
        verbose=False
):
    root = ET.parse(path).getroot()
    namespace = re.match(r"\{(.+)\}", root.tag)
    if namespace: namespace = "{" + namespace.group(1) + "}"
    d = root.find(f"{namespace}g").find(f"{namespace}path").attrib['d']
    p = svg.path.parse_path(d)
    m = [p.point(i/points) for i in range(points)]
    m = np.array([[c.real, c.imag] for c in m])
    return m

if __name__ == "__main__":
    path = extract_path_from_svg(verbose=True)
    print(path)
