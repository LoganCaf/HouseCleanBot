### File: map.py
# This file contains the Map class, which represents a grid-based environment for an agent to navigate.

import cv2 as cv
import numpy as np
from PIL import Image

class Map:
    def __init__(self, length, width,agentSize=10,MAXSIZE=400,MAXBANDS=15):
        self.length = length
        self.width = width
        self.agentSize = agentSize
        self.agent = None
        self.grid = np.zeros((MAXSIZE, MAXSIZE, MAXBANDS)).astype(np.float16)
        # grind bands:
        # if all 0 then assumed nothing is there
        # 0 - wall
        # 1 - agent
        # 2 - cleaned


    def add_wall(self, x1, y1, x2, y2):
        self.grid[x1:x2, y1:y2, 0] = 1 # set wall

    def add_agent(self, x, y):
        self.agent = (x, y)
        for r in range(-(self.agentSize//2), (self.agentSize//2)+1):
            for c in range(-(self.agentSize//2), (self.agentSize//2)+1):
                if x + r < self.length and y + c < self.width:
                    if self.grid[x + r,y + c,0] != 0:
                        raise ValueError("Invalid position for agent")
                    self.grid[x + r,y + c,1] = 1
    
    def remove_agent(self):
        for r in range(-(self.agentSize//2), (self.agentSize//2)+1):
            for c in range(-(self.agentSize//2), (self.agentSize//2)+1):
                if self.agent[0] + r < self.length and self.agent[1] + c < self.width:
                    self.grid[self.agent[0] + r,self.agent[1] + c,2] = 1 # set where i was to clean
                    self.grid[self.agent[0] + r,self.agent[1] + c,1] = 0 # remove agent
        self.agent = None
    
    def checkCollision(self, x, y):
        for r in range(-(self.agentSize//2), (self.agentSize//2)+1):
            for c in range(-(self.agentSize//2), (self.agentSize//2)+1):
                if x + r < self.length and y + c < self.width:
                    if self.grid[x + r,y + c,0] != 0:
                        return True
        return False

    def displayMove(self):
        while True:
            img = self.displayBase(False)
            key = cv.waitKey(1) & 0xFF
            if key == ord('w'):
                self.move_agent(self.agent[0] - 1, self.agent[1])
            elif key == ord('a'):
                self.move_agent(self.agent[0], self.agent[1] - 1)
            elif key == ord('s'):
                self.move_agent(self.agent[0] + 1, self.agent[1])
            elif key == ord('d'):
                self.move_agent(self.agent[0], self.agent[1] + 1)
            if key == ord('q'):
                break
            cv.imshow("Environment", img)
    
    def displayBase(self,show=True):
        cell_size = 5
        img = np.zeros((self.length * cell_size, self.width * cell_size, 3), np.uint8)
        for x in range(self.length):
            for y in range(self.width):
                color = [0, 0, 0]
                if self.grid[x,y,0] == 1:
                    color = [255, 255, 255]
                elif self.grid[x,y,1] == 1:
                    color = [0, 255, 0]
                elif self.grid[x,y,2] == 1:
                    color = [50, 50, 50]
                img[x * cell_size:(x + 1) * cell_size, y * cell_size:(y + 1) * cell_size] = color

        if show:
            cv.imshow("Environment", img)
            cv.waitKey(1)
        return img
    
    def close(self):
        cv.destroyAllWindows()

    def move_agent(self, x, y):
        if not self.checkCollision(x, y):
            self.remove_agent()
            self.add_agent(x, y)

    
    def move_direction(self, direction):
        mult = (((direction // 4)+1)*self.agentSize)//4 # movement speed
        direction = direction%4 #movement direction
        match direction:
            case 0:
                self.move_agent(self.agent[0] - mult, self.agent[1])
            case 1:
                self.move_agent(self.agent[0] + mult, self.agent[1])
            case 2:
                self.move_agent(self.agent[0], self.agent[1] - mult)
            case 3:
                self.move_agent(self.agent[0], self.agent[1] + mult)
    

    def getGrid3D(self):
        return self.grid

    def getMovableCount(self):
        count = 0
        for x in range(self.length):
            for y in range(self.width):
                if self.grid[x,y,0] != 1:
                    count += 1
        return count


        

if __name__ == "__main__":
    m = Map(10, 10)
    m.add_wall(0, 0, 10, 1)
    m.add_wall(0, 0, 1, 10)
    m.add_wall(9, 0, 10, 10)
    m.add_wall(0, 9, 10, 10)
    m.add_agent(5, 5)
    m.displayMove()
    m.close()

