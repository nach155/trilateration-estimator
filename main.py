import cvxopt
import numpy as np
from vehicle import Vehicle
from anchor import Anchor

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    vehicle = Vehicle(5,5,0)
    
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
    
    anchors = [Anchor(0,0),Anchor(7,0),Anchor(0,5),Anchor(10,10)]
    
    for i in range(314):
        vehicle.step(np.array([0.2,0.2]))
        for anchor in anchors:
            img = img + plt.plot(anchor.x,anchor.y,color="#000000",marker="o")
            distance = anchor.measure(np.array([vehicle.x,vehicle.y]))
            img = img + plt.plot(distance*np.cos(t)+anchor.x,distance*np.sin(t)+anchor.y,color="#CCCCCC")
        img = plt.plot(vehicle.x,vehicle.y,color="#ff00ff",marker="o")
        history.append(img)
        
    ani = animation.ArtistAnimation(fig,history,interval=50)
    plt.show()

#################################################
if __name__ == '__main__':
    main()