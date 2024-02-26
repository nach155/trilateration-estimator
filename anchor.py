import numpy as np

class Anchor(object):
    # 標準偏差
    d = 0.03
    
    def __init__(self, x:float,y:float,z:float)->None:
        self.x = x
        self.y = y
        self.z = z        
    
    def measure(self,position:np.array, nominal:bool=False)->float:
        noise = np.random.normal(0,self.d)
        if nominal:
            noise = 0   
        distance = np.linalg.norm(position - np.array([self.x,self.y,self.z]))*(1 + noise)
        return distance
    
    def state(self)->np.array:
        return np.array([[self.x],[self.y],[self.z]])
    
#################################################
if __name__ == '__main__':
    anchor = Anchor(1,1)
    measure = anchor.measure(np.array([2,2]))
    print(measure)