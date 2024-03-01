import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vehicle import Vehicle
from anchor import Anchor
from trilateration import TrilaterationEstimator
from filter import ParticleFilter

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
    
    variant = np.diag([0.05,0.06,0.2])
    init_mean = np.array([5,5,0.2])
    particle_filter = ParticleFilter(variant,init_mean,1000)
    
    for i in range(314):
        vehicle.step(np.array([0.2,0.2]))
        distance_list = []
        img = []
        for anchor in anchor_list:
            img = img + plt.plot(anchor.x,anchor.y,color="#BBBBBB",marker="o")
            distance = anchor.measure(vehicle.position())
            distance_list.append(distance)
        try:
            estimated_position = estimator.estimate(distance_list)
            filtered_position = particle_filter.estimate(estimated_position)
            particle_filter.resampling()
            # print(filtered_position)
        except Exception as e:
            continue

        for anchor,distance in zip(anchor_list, distance_list):
            distance_2d = np.sqrt(distance**2 - (anchor.z - vehicle.z)**2)
            img = img + plt.plot(distance_2d*np.cos(t)+anchor.x,distance_2d*np.sin(t)+anchor.y,color="#BBBBBB")
            
        # 履歴に追加
        img = img + plt.plot(particle_filter.particles[:,0],particle_filter.particles[:,1],color="#FF0000",marker='o',alpha=0.3,markersize=1,linestyle='None')
        img = img + plt.plot(vehicle.x,vehicle.y,color="#FF00FF",marker="o")
        img = img + plt.plot(estimated_position[0],estimated_position[1],color="#00FF00",marker='o')
        img = img + plt.plot(filtered_position[0],filtered_position[1],color="#00FFFF",marker='o')
        history.append(img)
        
    ani = animation.ArtistAnimation(fig,history,interval=100)
    plt.show()

#################################################
if __name__ == '__main__':
    main()