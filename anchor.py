import numpy as np

class Anchor(object):
    # 標準偏差
    d = 0.05
    
    def __init__(self, x:float,y:float)->None:
        self.x = x
        self.y = y
        
    def measure(self,position:np.array)->float:
        noise = np.random.normal(0,self.d)
        distance = np.linalg.norm(position - np.array([self.x,self.y])) + noise
        return distance
    
#################################################
if __name__ == '__main__':
    anchor = Anchor(1,1)
    measure = anchor.measure(np.array([2,2]))
    print(measure)