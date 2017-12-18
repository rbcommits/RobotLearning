import sys
import numpy as np
from math import sin, cos, radians

num_links = 5
def Translate(x, y, z):
  return np.array([
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1]])

rotation = [
    lambda angle: np.array([[1, 0, 0, 0],[0, cos(angle), -sin(angle), 0],[0, sin(angle), cos(angle), 0],[0, 0, 0, 1]]),
    lambda angle: np.array([[cos(angle), 0, sin(angle), 0],[0, 1, 0, 0],[-sin(angle), 0, cos(angle), 0],[0, 0, 0, 1]]),
    lambda angle: np.array([[cos(angle), -sin(angle), 0, 0],[sin(angle), cos(angle), 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
]
def get_rotations(lengths, angles):
    # convert all the angles into radians
    q =  [radians(theta) for theta in angles]
    #for angle in angles:
    #q.append(radians(angle))
    positions = []
    #homogeneous transformation
    for i in range(num_links):
        positions = [np.array([0, 0, 0, 1])] + positions
        k = num_links - i - 1
        for j in range(len(positions)):
            
            #if k == 9 or k == 9:
            #    positions[j] = np.dot(np.dot(Translate( lengths[k],0, 0), rotation[2](q[k])), positions[j])
            #elif True:
            #    positions[j] = np.dot(np.dot(Translate( lengths[k],0, 0), rotation[1](q[k])), positions[j])
            #else:
            positions[j] = np.dot(np.dot(Translate( -lengths[k], 0 ,0), rotation[2](q[k])), positions[j])


    return np.array(positions)[:,:3]

