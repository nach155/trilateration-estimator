import numpy as np
class Vehicle(object):
    # サンプリング時間
    ts = 0.1

    def __init__(self, x:float, y:float,z:float ,theta:float)->None:
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
    
    def step(self, input:np.array):
        self.x = self.x + input[0]*np.cos(self.theta)*self.ts
        self.y = self.y + input[0]*np.sin(self.theta)*self.ts
        self.theta = self.theta + self.ts * input[1]