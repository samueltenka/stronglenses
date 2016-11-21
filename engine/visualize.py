''' author: sam tenka
    date: 2016-11-20
    descr: Visualize vis-set images.
'''

from utils.config import get
from data_scrape.fetch_data import fetch_Xy
import numpy as np
import matplotlib.pyplot as plt

X_vis, y_vis = fetch_Xy('VIS') 

img = X_vis[0]
print(img.shape)
maxes = (np.amax(img[:,:,0]) for i in range(3))
mins = (np.amin(img[:,:,0]) for i in range(3))
plt.imshow(X_vis[0]) 
plt.show()
