from math import cos, tan, sqrt, exp, sin
import numpy as np
from forward_kinematics import get_rotations
import math
# pylint: disable=too-many-instance-attributes
class BallGame:
    # pylint: disable=too-many-instance-attributes
    def __init__(self):#[-5, 45, -10, -10, -5, -10, -5]
        self.base_state = np.concatenate([np.zeros([7], dtype='float') ,np.zeros([8], dtype='float')]) #store angles and velocities in 1 array
        self.link_lengths = np.array([0, 0, 1, 1, 0, 1, 1], dtype='float')
        self.state = self.base_state.copy()
        self.num_links = 6
        self.hoop = (5, 5, 0)
        self.g = -9.810
        self.episode_finished = False
        self.max_reward = 100
        self.hoop_radius = 0.2
        self.time_interval = 0.1
        self.time = 0
        self.action_range = np.array([(-180.0, 180.0), (-180.0, 180.0),(-180.0, 180.0),(-180.0, 180.0),(-180.0, 180.0),(-180.0, 180.0),(0.0, 15.0),(0.0, 1.0)])

    def get_state(self):
        return self.state
    def get_num_actions(self):
        return self.num_links + 1
    '''
    Given an action(velocity, theta), figures out if the action was
    sucessful and the ball passed through the hoop or not. If it is 
    unsucessful, finds out how far the ball was from the hoop
    '''
    def make_action(self, action):
        action = np.array(action, dtype='float')
        self.time += 1
        links = 7
        #print action
        throw = bool(action[-1])
        #action = action[:7]
        #action[action == 0] -= 1
        action *= -5                # go from -15 to 15 degrees
        reward = -100
        y=0
        next_state, new_pos = self.forward_kin(action)
        #make reward calculation
        #reward = self.calculate_reward(self.state, next_state)
        if(throw):
            reward, y = self.calculate_reward(abs(np.sum(next_state[:links-1])), next_state[-2], new_pos[-1])
            self.episode_finished = True
        elif(self.time >= 10):
            reward, y = self.calculate_reward(abs(np.sum(next_state[:links-1])), next_state[-2], new_pos[-1])
            self.episode_finished = True
        self.state = next_state
        #print(reward)
        return reward, y
    

    def forward_kin(self, action):
        time = 0.5
        links = 7
        # solve for the next position and velocity
        next_state = self.state.copy()
        # calculate new angles
        next_state[:links] += action[:links]
        initial_pos = get_rotations(self.link_lengths, self.state[:links])
        new_pos = get_rotations(self.link_lengths, next_state[:links])
        #for i in range(links):
        #    next_state[i] += np.sum(action[:(i+1)])
        

        velocities = np.absolute(np.absolute(new_pos) - np.absolute(initial_pos)) / time
        velocity = (velocities[:,0]*np.cos(next_state[:7])) + np.absolute( velocities[:,1]*np.sin(next_state[:7]))
        next_state[links:-1] += velocity[:]
        stateRange = np.array(self.state_range(), dtype='float')
        next_state = np.minimum(
            np.maximum(next_state, stateRange[:, 0]), stateRange[:, 1])
        return next_state, new_pos
        #reward = self.calculate_reward(np.sum(next_state[:links-1]), next_state[-2], new_pos[-1])
        '''
        time = 0.1
        A = np.concatenate([np.eye(7), np.eye(7) * t, np.zeros([7, 1])], axis=1)
        # solve for the next position and velocity
        next_state = self.state.copy()
        # calculate new angles
        for i in range(7):
            next_state[0] = np.sum(action[:(i+1)])
        initial_pos = get_rotations(self.link_lengths, self.state[:7])
        new_pos = get_rotations(self.link_lengths, next_state[:7])

        # now get velocities

        next_state[7:] = action
        next_state[:7] = (np.dot(A, next_state.T).T)[0]
        stateRange = np.array(self.state_range(), dtype='float')
        next_state = np.minimum(
            np.maximum(next_state, stateRange[:, 0]), stateRange[:, 1])
        #stateRange = np.array(self.state_range(), dtype=np.float32)
        #nextState = np.minimum(
        #    np.maximum(nextState, stateRange[:, 0]), stateRange[:, 1])
        # find reward and termination
        #done = self.terminationFn(nextState, action)
        return next_state
        '''
    

    '''
    https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Trajectory_of_a_projectile.html
    use that for more information about the equation used
    '''
    def calculate_reward(self, angle, velocity, launch_pos):
        g = -9.8
        pos = self.hoop[1] - launch_pos[1]
        y = np.absolute( pos*math.tan(angle) - ( ( g*pos*pos )/( 2*velocity*velocity*0.5*(1-math.cos(2*angle)) ) ) )
        threshold = 0.2 # essentially the radius of the ball
        #print("%f %f" % ( np.exp( -(self.hoop[0] - y + threshold)**2), np.exp( -(self.hoop[0] - y - threshold)**2) ))
        reward = np.max([np.exp( -(self.hoop[0] - y + threshold)**2), np.exp( -(self.hoop[0] - y - threshold)**2)] ) * 100
        if reward <= 0.01:
            reward = -100
        return reward, y

        '''
        pos = get_rotations(self.link_lengths, state[:7])[-1, :]
        p0 = get_rotations(self.link_lengths, state[:7])
        p1 = get_rotations(self.link_lengths, next_state[:7])
        vel = (p1[-1,:3] - p0[-1,:3]) / 0.01
        vz = vel[2]
        dz = pos[2] - self.hoop[2]
        # compute the time it would take to reach the goal (kinematics equations)
        '''
        '''
        g = -4.9
        
        
        # quadratic formula (+/- sqrt)
        # the parabola doesn't even reach the line
        b2_4ac = vz * vz - 4 * g * dz
        if b2_4ac < 0:
            b2_4ac = b2_4ac**2
            #return 0.0
        dt1 = (-vz + sqrt(b2_4ac)) / (2 * g)
        dt2 = (-vz - sqrt(b2_4ac)) / (2 * g)
        dt = max(dt1, dt2)
        # the ball would have to go backwards in time to reach the goal
        #if dt < 0:
        #    return 0.0

        # find the distance from the goal (in the xy-plane) that the ball has hit
        dp = self.hoop[:2] - (pos[:2] + vel[:2] * dt)
        # use a kernel distance
        '''

        #=================================================================================================
        '''
        def x(t):
            return velocity*math.cos(angle - 90)*t + launch_pos[0]
        def y(t):
        '''

        '''
        a = (2*(velocity*cos(angle))**2)
        b = tan(angle)
        c = -self.hoop[1] + 0
        y = abs((self.hoop[0]*b - (self.g*self.hoop[0]**2)/a ) - self.hoop[1])

        b2_4ac = b**2 - 4*a*c
        if b2_4ac < 0:
            return 0.0


        x1 = abs( (-b + sqrt(b2_4ac))/2*a - self.hoop[0] )
        x2 = abs( (-b - sqrt(b2_4ac))/2*a - self.hoop[0] )
        
        x = min(x1, x2)
        dp = sqrt( (x-self.hoop[0])**2 + ( y-self.hoop[1] )**2 )
        #print(dp)
        print("%f %f | %f %f " % (x, y, velocity, theta))
        print(state)
        print("============================================")
        '''


    def new_episode(self):
        self.state = self.base_state.copy()
        self.episode_finished = False
        self.time = 0 

    def is_episode_finished(self):
        # find out if trial is finished and update episode_finished variable
        return self.episode_finished

    def state_range(self):
        return [(-180.0, 180.0),  # angles
                (-180.0, 180.0),
                (-180.0, 180.0),
                (-180.0, 180.0),
                (-180.0, 180.0),
                (-180.0, 180.0),
                (0, 90),
                (-180.0, 180.0),  # velocities (max RPM)
                (-180.0, 180.0),
                (-180.0, 180.0),
                (-180.0, 180.0),
                (-180.0, 180.0),
                (-180.0, 180.0),
                (0, 20.0),
                (0.0, 1.0)]       # release