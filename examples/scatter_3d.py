__author__ = 'farrukh'

import matplotlib.pyplot as plt
from numpy.random import random
from mpl_toolkits.mplot3d import Axes3D

colors=['b', 'c', 'y', 'm', 'pink']

ax = plt.subplot(111, projection='3d')

ax.plot(random(10), random(10), random(10), 'x', color=colors[0], label='Low Outlier')
ax.plot(random(10), random(10), random(10), 'o', color=colors[0], label='LoLo')
ax.plot(random(10), random(10), random(10), 'o', color=colors[1], label='Lo')
ax.plot(random(10), random(10), random(10), 'o', color=colors[2], label='Average')
ax.plot(random(10), random(10), random(10), 'o', color=colors[3], label='Hi')
ax.plot(random(10), random(10), random(10), 'o', color=colors[4], label='HiHi')
ax.plot(random(10), random(10), random(10), 'x', color=colors[4], label='High Outlier')
plt.grid(True, linestyle='-', color='0.75')
plt.xlabel("X")
plt.ylabel("Y")
#plt.zlabel("Z")

plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

plt.show()