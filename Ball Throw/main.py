import argparse
import itertools as it
import numpy as np

#import tensorflow as tf
import os
import keras

from random import sample, random, randint
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape, Flatten
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.optimizers import RMSprop
#from time import sleep
from tqdm import trange, tqdm
from environment import BallGame


# neural network variables
learning_rate = 25e-5
epochs = 100
batch_size = 256                     # 256 seems optimal for my 2 gig GPU
memory_size = 20000                 # number of entries to store in it's memory. Seems rather low, could increase it as we proceed
discount_factor = 0.90
collection_per_epoch = 1000
testing_per_epoch = 10
save_model = True
save_interval = 10000                # number of training episodes after which the model is saved to the disk. I use a lowe number since my gPU is slow and I usually quit prematurely
load_model = True
epsilon = 1.0
observe_episodes = 1000
memory = []
model_savefile_location = "models/Jarvis-{0}-{1}".format(5, learning_rate)
#model_state_shape = (1, image_resolution[0], image_resolution[1], channels)
collect_data_bool = False
train_data_bool = False
test_data_bool = False
train_and_collect = False
# other functional variables
jarvis_config = "jarvis.cfg"
num_links = 6
trained_data = 0


def initialize_game():
    print("Initializing Robot Engine.....")
    # add basic unity init stuff here for root
    '''
    game = DoomGame()
    print "Loading config file {0}".format(config_file_path)
    game.load_config(config_file_path)
    print "config file loaded successfully...."
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print "Doom Engine initialized"
    '''
    game = BallGame()
    return game


def initialze_model(num_output, num_input):
    print("Initializing JARVIS....")
    ''' trying to follow the deep mind structure.
        - Layer 1: Dense,             Activation = Relu,   output = 32, strides = 4, Kernal size 8x8
        - Layer 2: Dense,             Activation = Relu,   output = 64, strides = 2, Kernal size 4x4
        - Layer 3: Dense,             Activation = Relu,   output = 64, strides = 1, Kernal size 3x3
        - Layer 5: Fully Conencted/Dense,   Activation = Relu,   output = 512,
        - Layer 6: Fully Conencted/Dense,   Activation = Linear, output = (1x1xnum_actions)
    '''

    model = Sequential()

    model.add(Dense(units=128, activation='relu', input_shape=num_input, bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dense(units=256, activation='relu', bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Conv1D(64,3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128, activation='relu', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
    model.add(Dense(units=512, activation='relu', bias_initializer=keras.initializers.Constant(0.1)))
    
    #model.add(Dense(units=64, activation='relu', bias_initializer=keras.initializers.Constant(0.1)))
    #model.add(Flatten())
    model.add(Dense(units=num_output, activation='relu', bias_initializer=keras.initializers.Constant(0.1)))
    
    #model.add(Dense(units=2, activation='linear', bias_initializer=keras.initializers.Constant(0.1)))
    #model.add(Reshape([8, 2]))
    rms_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='mse',
                  optimizer=rms_optimizer, metrics=['acc'])

    #if load_model:
    #    print("Loading weights....")
    #    model.load_weights(model_savefile_location, by_name=True)
    print("Initialization finished....")
    print("outputs ", num_output)
    return model


def get_action(state):
    state = state.reshape([1, -1, 1])
    return np.argmax((model.predict(state, batch_size=1, verbose=0)), axis=1)


def get_q_values(state):
    state = state.reshape([state.shape[0], -1, 1])
    return (model.predict(state, batch_size=state.shape[0], verbose=0))


def get_reward():
    # simulate ball throw here
    reward = 0
    return reward


def get_random_action():
    # return np.random.randint(2, size=[8])
    return randint(0, len(actions)-1)


def collect_data(game, model, epoch=1):
    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    state1 = game.get_state()
    action = 0
    if random() <= exploration_rate(epoch):  # or training_episodes_finished < observe_episodes:
        action = get_random_action()  # make a completly random action
    else:
        action = get_action(state1)[0]  # np.argmax(model.predict(image, batch_size=1),1) # get the best out of the prediction
    reward, _ = game.make_action(actions[action])
    state2 = game.get_state()

    memory.append([state1, action, reward, state2])  # here we use the image to describe state. we can add other parameters later as well!
    
    if len(memory) > memory_size:
        memory.pop(0)  # if we exceed, remove old memories


def train_model(game, model, memory):
    while len(memory) < batch_size:
        collect_data(game, model)

    batch = sample(memory, batch_size)
    # s1, a, r, s2 = batch
    # print(batch)
    s1 = np.array([i[0] for i in batch])
    a = np.array([i[1] for i in batch])
    r = np.array([i[2] for i in batch])
    s2 = np.array([i[3] for i in batch])

    q1 = get_q_values(s1)
    q2 = np.max(get_q_values(s2), axis=1)

    # is_discounted = np.array((r>0), dtype=np.int32).reshape(len(batch),)
    #print s1_Q.shape
    #print r.shape
    #print q2.shape
    #if(r > 0.5):
    q1[np.arange(q1.shape[0]), a] += r + discount_factor * q2  # * is_discounted  # Update q value with Bellman's equation if we have positive reward, otherwise just add reward
    #else:
    #    q1[np.arange(q1.shape[0]), a] = r + discount_factor * q2 - 100
    #for i in range(s1_Q.shape[0]):
    #    s1_Q[i]
    s1 = s1.reshape([s1.shape[0], -1, 1])
    model.fit(s1, q1, batch_size=batch_size, epochs=20, verbose=0)
    return batch_size  # to keep track of how many memories we have trained upon


def test_model(game, model):
    test_phase_rewards = []
    #tqdm.write("Testing Model....")
    for episode in range(5):
        game.new_episode()
        # tqdm.write("Episode {0}....".format(episode+1))
        reward = 123
        while not game.is_episode_finished():
            state = game.get_state()
            index = get_action(state)[0]  # get the best out of the prediction
            reward, y = game.make_action(actions[index])
            if reward == 0:
                print(y)
        test_phase_rewards.append( reward ) #game.get_total_reward())
    return test_phase_rewards



''' BEGIN Argument Parsing '''

# Parse arguments for location of cfg file and wad file. Otherwise use defaults
parser = argparse.ArgumentParser()
parser.add_argument("-c", action="store_true", help="If you want to collect data for training")
parser.add_argument("-t", action="store_true", help="if you want to train on previously collected data")
parser.add_argument("-te", action="store_true", help="if you want to test your model playing")
parser.add_argument("-l", action="store_true", help="if you want to load a previously saved model")
parser.add_argument("-e", help="number of epochs")
parser.add_argument("-tac", action="store_true", help="if you want to perform training and collection at the same time")
args = parser.parse_args(args=None, namespace=None)

if args.c:
    collect_data_bool = True
    print("Data collection enabled")
if args.t:
    train_data_bool = True
    print("Model training enabled")
if args.te:
    test_data_bool = True
    print("Model Testing enabled")
if args.l:
    load_model = True
else:
    load_model = False
if args.e:
    epochs = int(args.e)
if args.tac:
    train_and_collect = True
# add support for other arguments later. Like config file attributes and neural network parameters

''' END Arument Parsing'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to disable tensorflow debugging. It gets rather annoying
game = initialize_game()
global num_actions
num_actions = game.get_num_actions()
global actions
actions = np.array([list(a) for a in it.product([0, 1], repeat=8)])
#actions = sample(actions, 63)
#actions.append([1,1,1,1,1,1,1,1])
model = initialze_model(len(actions), (15,1))
model.summary()

def final_train(game, model, memory):
    s1 = np.array([i[0] for i in memory])
    a = np.array([i[1] for i in memory])
    r = np.array([i[2] for i in memory])
    s2 = np.array([i[3] for i in memory])
    print(s1)
    s1_Q = get_q_values(s1)
    q2 = np.max(get_q_values(s2), axis=1)
    # is_discounted = np.array((r>0), dtype=np.int32).reshape(len(batch),)
    #print s1_Q.shape
    #print r.shape
    #print q2.shape
    s1_Q[np.arange(s1_Q.shape[0]), a] = r + discount_factor * q2  # * is_discounted  # Update q value with Bellman's equation if we have positive reward, otherwise just add reward
    #for i in range(s1_Q.shape[0]):
    #    s1_Q[i]
    model.fit(s1, s1_Q, batch_size=batch_size, epochs=10, verbose=1)


current_episode = 0
for epoch in range(epochs):
    #print("\n=============================\nEpoch {0}....\n=============================\n".format(epoch + 1))
    print("Episode: %d" % (current_episode))
   
    global_rewards = []

    
    #tqdm.write("\tStarting learning episode")
    game.new_episode()
    for i in range(collection_per_epoch):
        collect_data(game, model, epoch)
    train_model(game, model, memory)
    if game.is_episode_finished():
        game.new_episode()
    #if current_episode % int(collection_per_epoch / 2) == 0:
        #print("\tSaving model {0}".format(model_savefile_location))
        # model.save_weights(model_savefile_location, overwrite=True)
    current_episode += 1
    total_rewards = np.array(test_model(game, model))

    print("Max reward {0}, Min reward: {1}, Mean reward {2}\n".format(total_rewards.max(), total_rewards.min(), total_rewards.mean()))


    


#final_train(game, model, memory)
game.new_episode()
state = game.get_state()
a = get_action(state)
print(actions[a])