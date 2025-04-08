import xml.etree.ElementTree as ET
import numpy as np

def parseSVG(svg_path, grid_size=(50, 50), canvas_size=(500, 500)):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {'svg':'http://www.w3.org/2000/svg'}     # Supposedly this is necessary 

    grid = np.zeros(grid_size, dtype=np.uint8)
    rows, cols = grid_size
    width, height = canvas_size

    def toGrid(x, y):
        gx = int((x / width) * cols)
        gy = int((y / height) * rows)
        return gy, gx
    
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]
        if tag == 'rect':
            x = float(elem.attrib.get('x', 0))
            y = float(elem.attrib.get('y', 0))
            w = float(elem.attrib.get('width', 0))
            h = float(elem.attrib.get('height', 0))

            top_left = toGrid(x, y)
            bottom_right = toGrid(x + w, y + h)

            for i in range(top_left[0], bottom_right[0]):
                for j in range(top_left[1], bottom_right[1]):
                    if 0 <= i < rows and 0 <= j < cols:
                        grid[i][j] = 1

    return grid

import matplotlib.pyplot as plt

grid = parseSVG('train-00/0000-0003.svg')
plt.imshow(grid, cmap='gray')
plt.title("Parsed Grid")
plt.show()
