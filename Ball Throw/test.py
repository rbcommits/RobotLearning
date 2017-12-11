
import numpy as np
from forward_kinematics import get_rotations
import math
links = 7
global base_state
base_state = np.concatenate([np.zeros([links], dtype='float') ,np.zeros([8], dtype='float')])
link_lengths = np.array([1 for i in range(links) ], dtype='float')
next_state = np.concatenate([np.zeros([links], dtype='float') ,np.zeros([8], dtype='float')])
global angles

def make_action(action):
    action = np.array(action, dtype='float')
    time = 1
    #print action
    action *= -5               # go from -15 to 15 degrees
    reward = 0
    forward_kin(action)
    #make reward calculation

    

def forward_kin(action):
    global base_state
    time = 0.3
    # solve for the next position and velocity
    next_state = base_state.copy()
    # calculate new angles
    next_state[:links] += action[:links]
    initial_pos = get_rotations(link_lengths, base_state[:links])
    new_pos = get_rotations(link_lengths, next_state[:links])
    #print("============")
    #print(initial_pos)
    #print(new_pos)
    #print("==============")

    #for i in range(links):
    #    next_state[i] = np.sum(action[:(i+1)])
    

    velocities = np.absolute(np.absolute(new_pos) - np.absolute(initial_pos)) / time
    angle = getAngle(new_pos[0], new_pos[-1])
    velocity = np.absolute(velocities[:,0]*math.cos(angle)) + np.absolute( velocities[:,1]*math.sin(angle))
    #next_state[links:-1] = velocity[:]
    #print(next_state)
    print(velocities)
    #print("==========")
    next_state[links:-1] += velocity[:]
    #print(next_state)
    base_state = next_state
    #print("==========")
    #print(initial_pos)
    print(new_pos)
    print(next_state)
    #print(velocity)
    print(angle)
    # now get velocities
    print("=================================================")

def getAngle(a, b):
    opp = b[1] - a[1]
    ad = b[0] - a[0]
    return math.degrees(math.atan(opp/ad)) 
def main():
    a1=[1,1,1,1,1,1,1]
    a2 =  [1 for i in range(links)]# #[1,1,1,1,1,1,1,0]
    a3=[1,0,0,0,0,0,0,0]
    make_action(a1)
    make_action(a1)
    make_action(a1)
    #make_action(a1)

if __name__ == '__main__':
    main()
    #print(getAngle([-1, 1], [-2,0]))