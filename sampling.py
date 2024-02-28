import numpy as np

from anchor import Anchor
from vehicle import Vehicle
from trilateration import TrilaterationEstimator

"""サンプリング
    サンプリングして分散共分散行列と平均を求めます
"""

def main():
    anchor_list = [
        Anchor(-0.1,0.1,2),
        Anchor(7,0,2.1),
        Anchor(0,5,1.9),
        Anchor(9.9,9.8,2),
        ]
    estimator = TrilaterationEstimator(anchor_list)
    error_history = np.array([[],[],[]])
    for i in range(10000):
        position = np.random.uniform(0,10,2)
        vehicle = Vehicle(position[0],position[1],0.2,0)
        distance_list = []
        for anchor in anchor_list:
            distance_list.append(anchor.measure(vehicle.position()))
        
        try:
            estimated_position = estimator.estimate(distance_list)
        except Exception as e:
            continue
        nominal_position = vehicle.position()
        error_position = estimated_position - nominal_position
        error_history = np.hstack((error_history,error_position))
    covariant = np.cov(error_history)
    mean = np.mean(error_history,axis=1)
    
    print(covariant)
    print(mean)
    print(error_history.shape)
        
#################################################
if __name__ == '__main__':
    main()
    