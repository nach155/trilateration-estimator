import numpy as np
from anchor import Anchor
    
class TrilaterationEstimator(object):
    def __init__(self,anchor_list:list[Anchor])->None:
        self.anchor_list = anchor_list
        self.N = len(anchor_list)
        self.__calc_c__()
        self.__calc_H__()
        
    def __calc_a__(self,distance_list:list[float])->np.ndarray:
        a = np.zeros(())
        for anchor, distance in zip(self.anchor_list, distance_list):
            anchor_position = anchor.position
            a = a + (anchor_position @ anchor_position.T @ anchor_position) - (distance**2 * anchor_position)
        a = a / self.N
        self.a = a
        return a
    
    def __calc_B__(self,distance_list)->np.ndarray:
        B = np.zeros(())
        for anchor, distance in zip(self.anchor_list,distance_list):
            anchor_position = anchor.position()
            B = B - 2*anchor_position@anchor_position.T - (anchor_position.T@anchor_position)*np.eye(3) + distance**2*np.eye(3)
        B = B / self.N
        self.B = B
        return B
    
    def __calc_a_B__(self,distance_list)->tuple[np.ndarray,np.ndarray]:
        a = np.zeros(())
        B = np.zeros(())
        for anchor, distance in zip(self.anchor_list,distance_list):
            anchor_position = anchor.position
            a = a + (anchor_position @ anchor_position.T @ anchor_position) - (distance**2 * anchor_position)
            B = B - 2*anchor_position@anchor_position.T - (anchor_position.T@anchor_position)*np.eye(3) + distance**2*np.eye(3)
        self.a = a / self.N
        self.B = B / self.N
        return (self.a, self.B)

    def __calc_c__(self)->np.ndarray:
        self.c = sum([anchor.position for anchor in self.anchor_list])/self.N
        return self.c

    def __calc_f__(self)->np.ndarray:
        self.f = self.a  + self.B @ self.c + 2 * self.c @ self.c.T @ self.c
        return self.f
    
    def __calc_D__(self)->np.ndarray:
        self.D = self.B + 2 * self.c @ self.c.T + (self.c.T @ self.c) * np.eye(3)
        return self.D
    
    def __calc_H__(self)->np.ndarray:
        # self._calc_D()
        h = np.zeros(())
        for anchor in self.anchor_list:
            anchor_position = anchor.position
            h = h + anchor_position@anchor_position.T
        self.H = - 2 * h / self.N + 2 * self.c @ self.c.T
        return self.H
    
    def __calc_position__(self,distance_list:list[float])->np.ndarray:
        f_d = (self.f - self.f[-1])[:-1]
        H_d = (self.H - self.H[-1])[:-1]
        
        (Q,U) = np.linalg.qr(H_d)
        V = Q.T @ f_d
        
        pr = 0
        for anchor, distance in zip(self.anchor_list, distance_list):
            pr = pr - anchor.position.T @ anchor.position + distance**2
        qTq = np.squeeze(pr / self.N + self.c.T @ self.c)
        
        j = np.squeeze((U[0,1]*V[1])/(U[0,0]*U[1,1]) - (V[0]/U[0,0]))
        k = np.squeeze((U[0,1]*U[1,2])/(U[0,0]*U[1,1]) - (U[0,2]/U[0,0]))
        
        l = np.squeeze(V[1]/U[1,1])
        m = np.squeeze(U[1,2]/U[1,1])
        
        a = k**2 + m**2 + 1
        b = j*k + l*m
        c = j**2 + l**2 - qTq
        
        D = b**2 - a*c
        if D < 0:
            raise TrilaterationException("Could not calculate q3.")
        if abs(D) <= 0.00001:
            D = 0
        q3_m = (-b - np.sqrt(D))/a
        # q3_p = (-b + np.sqrt(D))/a
        
        q_m = np.array([[j + k*q3_m],[-l-m*q3_m],[q3_m]])
        # q_p = np.array([[j + k*q3_p],[-l-m*q3_p],[q3_p]])
        
        return q_m + self.c
    
    def estimate(self,distance_list)->np.ndarray:
        self.__calc_a_B__(distance_list)
        self.__calc_f__()
        # if np.linalg.det(self.H) == 0:
        #     return self.__calc_position__(distance_list)
        # else:
        #     return - np.linalg.inv(self.H) @ self.f + self.c
        return self.__calc_position__(distance_list)
    
class TrilaterationException(Exception):
    pass
#################################################
if __name__ == '__main__':
    estimator = TrilaterationEstimator([Anchor(0,2,3),Anchor(3,0,3),Anchor(5,4,3),Anchor(0,0,3)])
    a = estimator.estimate([2.21*1.4,3.2,2.3,1.5*3])
    # print(a)