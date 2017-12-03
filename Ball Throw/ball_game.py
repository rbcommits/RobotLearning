from math import cos, sin, tan, sqrt, exp
import math
class game:
    hoop = (0.0, 0.0)
    state = []
    g = -9.81
    episode_finished = False
    y0=0
    max_reward = 100
    hoop_radius = 0.2


    def _init_(self):
        self.state = []

    
    def get_state(self):
        return self.state
    '''
    Given an action(velocity, theta), figures out if the action was
    sucessful and the ball passed through the hoop or not. If it is 
    unsucessful, finds out how far the ball was from the hoop
    '''
    def make_action(self, action, throw):
        reward = 0
        state, velocity, theta = self.forward_kin(action)
        #make reward calculation
        if(throw):
            reward = self.calculate_reward(velocity, theta)
            self.episode_finished = True
        return state, reward
    

    def forward_kin(self, action):
        state = []
        theta = 0
        velocity = 0
        return state, velocity, theta
    

    '''
    https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Trajectory_of_a_projectile.html
    use that for more information about the equation used
    '''
    def calculate_reward(self, velocity, theta):
        def in_hoop_y(y):
            return y >= (self.hoop[1] - self.hoop_radius) and y <= (self.hoop[1] + self.hoop_radius)
        def in_hoop_x(x):
            return x >= (self.hoop[0] - self.hoop_radius) and x <= (self.hoop[0] + self.hoop_radius)
        # a, b, c of the quadratic eq
        a = (2*(velocity*cos(theta))**2)
        b = tan(theta)
        c = -self.hoop[1] + self.y0
        y = abs(( self.y0 + self.hoop[0]*b - (self.g*self.hoop[0]**2)/a ) - self.hoop[1])
        x1 = 99999.0
        x2 = 99999.0


        try:
            x1 = abs( (-b + sqrt(b**2 - 4*a*c))/2*a - self.hoop[0] )
        except:
            # error happens when imaginary roots occur
            # we simply ignore them
            pass
        try:
            x2 = abs( (-b - sqrt(b**2 - 4*a*c))/2*a - self.hoop[0] )
        except:
            # error happens when imaginary roots occur
            # we simply ignore them
            pass
        if(x2 > x1):
            x1 = x2 
        return exp( -x1 )*self.max_reward + exp( -y )*self.max_reward #reward is inverse of distance

    def init_game(self):
        # make everything 0 or whatever. restart the game
        self.state = []

    def new_episode(self):
        self.init_game() 

    def is_episode_finished(self):
        # find out if trial is finished and update episode_finished variable
        return self.episode_finished
        
