import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vehicle import Vehicle
from anchor import Anchor
from trilateration import TrilaterationEstimator

def main():
    vehicle = Vehicle(5,5,0.2,0)
    
    # グラフの設定
    fig = plt.figure()
    plt.axis('scaled')
    plt.ylim(0,10)
    plt.xlim(0,10)
    plt.grid()
    history  = []
    t = np.linspace(0,2*np.pi,100)
    
    # 初期状態
    img = plt.plot(vehicle.x,vehicle.y,color="#ff00ff",marker="o")
    history.append(img)
    
    anchor_list = [
        Anchor(0,0,3),
        Anchor(7,0,3),
        Anchor(0,5,3),
        Anchor(10,10,3),
        ]
    
    estimator = TrilaterationEstimator(anchor_list)
    # 二次計画法の重み設定
    n = len(anchor_list)
    
    
    for i in range(314):
        vehicle.step(np.array([0.2,0.2]))
        distance_list = []
        img = []
        for anchor in anchor_list:
            img = img + plt.plot(anchor.x,anchor.y,color="#000000",marker="o")
            distance = anchor.measure(vehicle.position())
            distance_list.append(distance)
        
        estimated_position = estimator.estimate(distance_list)

        for anchor,distance in zip(anchor_list, distance_list):
            distance_2d = np.sqrt(distance**2 - (anchor.z - vehicle.z)**2)
            img = img + plt.plot(distance_2d*np.cos(t)+anchor.x,distance_2d*np.sin(t)+anchor.y,color="#0000FF")
            
        # 履歴に追加
        img = img + plt.plot(vehicle.x,vehicle.y,color="#FF00FF",marker="o")
        img = img + plt.plot(estimated_position[0],estimated_position[1],color="#00FF00",marker='o')
        history.append(img)
        
    ani = animation.ArtistAnimation(fig,history,interval=50)
    plt.show()

#################################################
if __name__ == '__main__':
    main()