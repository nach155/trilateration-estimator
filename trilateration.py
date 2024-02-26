import numpy as np
from anchor import Anchor
    
class TrilaterationEstimator(object):
    def __init__(self,anchor_list:list)->None:
        self.anchor_list = anchor_list
        self.N = len(anchor_list)
        
    def _calc_a(self,distance_list:list)->np.array:
        a = np.zeros((3,1))
        for anchor, distance in zip(self.anchor_list, distance_list):
            anchor_position = anchor.state()
            a = a + (anchor_position @ anchor_position.T @ anchor_position) - (distance**2 * anchor_position)
        a = a / self.N
        self.a = a
        return a
    
    def _calc_B(self,distance_list)->np.array:
        B = np.zeros((3,3))
        for anchor, distance in zip(self.anchor_list,distance_list):
            anchor_position = anchor.state()
            B = B - 2*anchor_position@anchor_position.T - (anchor_position.T@anchor_position)*np.eye(3) + distance**2*np.eye(3)
        B = B / self.N
        self.B = B
        return B
    
    def _calc_c(self)->np.array:
        self.c = sum([anchor.state() for anchor in self.anchor_list])/self.N
        return self.c

    def _calc_f(self)->np.array:
        self.f = self.a  + self.B @ self.c + 2 * self.c @ self.c.T @ self.c
        return self.f
    
    def _calc_D(self)->np.array:
        self.D = self.B + 2 * self.c @ self.c.T + (self.c.T @ self.c) * np.eye(3)
        return self.D
    
    def _calc_H(self)->np.array:
        # self._calc_D()
        h = np.zeros((3,3))
        for anchor in self.anchor_list:
            anchor_position = anchor.state()
            h = h + anchor_position@anchor_position.T
        self.H = - 2 * h / self.N + 2 * self.c @ self.c.T
        return self.H
    
    def estimate(self,distance_list):
        self._calc_a(distance_list)
        self._calc_B(distance_list)
        self._calc_c()
        self._calc_f()
        self._calc_H()
        print(self.H)
        print(np.linalg.det(self.H))
#################################################
if __name__ == '__main__':
    estimator = TrilaterationEstimator([Anchor(0,2,3),Anchor(3,0,3),Anchor(5,4,3),Anchor(0,0,3)])
    a = estimator.estimate([2.21*1.4,3.2,2.3,1.5*3])
    # print(a)