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
# from tqdm import trange, tqdm
from environment import BallGame
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt


# neural network variables
learning_rate = 25e-5
epochs = 100
batch_size = 128                     # 256 seems optimal for my 2 gig GPU
memory_size = 20000                 # number of entries to store in it's memory. Seems rather low, could increase it as we proceed
discount_factor = 0.95
collection_per_epoch = 1000
testing_per_epoch = 5
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
jarvis_config = "game.cfg"
num_links = 6
trained_data = 0

tensorboard = TensorBoard(log_dir="Graph/")

def initialize_game():
    print("Initializing Robot Engine.....")
    # add basic unity init stuff here for root
    game = BallGame()
    return game


def initialze_model(num_output, num_input):
    print("Initializing Model....")

    model = Sequential()

    model.add(Dense(units=128, activation='relu', input_shape=num_input, bias_initializer=keras.initializers.Constant(0.01)))
    model.add(Dense(units=256, activation='relu', bias_initializer=keras.initializers.Constant(0.01)))
    #model.add(Conv1D(64,3, activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(LSTM(128, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='ones', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=True, unroll=False))
    model.add(Dense(units=256, activation='relu', bias_initializer=keras.initializers.Constant(0.01)))
    
    #model.add(Dense(units=64, activation='relu', bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Flatten())
    model.add(Dense(units=num_output, activation='linear', bias_initializer=keras.initializers.Constant(0.01)))
    
    #model.add(Dense(units=2, activation='linear', bias_initializer=keras.initializers.Constant(0.1)))
    #model.add(Reshape([8, 2]))
    rms_optimizer = RMSprop(lr=learning_rate)
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.5, decay=0.001, nesterov=True)
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.995, epsilon=1e-8)
    model.compile(loss='mse',
                  optimizer=adam, metrics=['acc'])
    '''
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=num_input, bias_initializer=keras.initializers.Constant(0.01)))
    model.add(Dense(units=256, activation='tanh', bias_initializer=keras.initializers.Constant(0.01)))
    model.add(Dense(units=256, activation='linear', bias_initializer=keras.initializers.Constant(0.01)))
    model.add(Dense(units=128, activation='relu', bias_initializer=keras.initializers.Constant(0.01)))
    model.add(Flatten())
    model.add(Dense(units=num_output, activation='tanh', bias_initializer=keras.initializers.Constant(0.01)))
    


    model.add(Dense(units=128, activation='linear', input_shape=num_input, bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dense(units=256, activation='linear', bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Conv1D(64,3, activation='linear'))
    model.add(MaxPooling1D(pool_size=5))
    #model.add(LSTM(128, activation='linear', recurrent_activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False))
    #model.add(LSTM(128))
    #model.add(Dense(units=256, activation='tanh', bias_initializer=keras.initializers.Constant(0.1)))
    
    model.add(Dense(units=128, activation='linear', bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Flatten())
    model.add(Dense(units=num_output, activation='linear', bias_initializer=keras.initializers.Constant(0.1)))
    
    #model.add(Dense(units=2, activation='linear', bias_initializer=keras.initializers.Constant(0.1)))
    #model.add(Reshape([8, 2]))
    '''
    #rms_optimizer = RMSprop(lr=learning_rate)
    #sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=True)
    #model.compile(loss='mae',
    #              optimizer='adam', metrics=['mae', 'acc'])

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


def collect_data(game, model, epoch):
    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.10
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
    game.new_episode()

    reward = 0
    while not game.is_episode_finished():

        state1 = game.get_state()
        action = 0
        if random() <= exploration_rate(epoch):  # or training_episodes_finished < observe_episodes:
            action = get_random_action()  # make a completly random action
        else:
            action = get_action(state1)[0]  # np.argmax(model.predict(image, batch_size=1),1) # get the best out of the prediction
        reward = game.make_action(actions[action], epoch)
        state2 = game.get_state()
        is_terminal = game.is_episode_finished()
        #temp_mem.append([state1, action, reward, state2, is_terminal])
        memory.append([state1, action, reward, state2, is_terminal])  # here we use the image to describe state. we can add other parameters later as well!
    #temp_mem = np.array(temp_mem)
    #temp_mem[:,2] = reward
    #for mem in temp_mem:
    #    memory.append(mem)
    if len(memory) > memory_size:
        memory.pop(0)  # if we exceed, remove old memories


def train_model(game, model, memory):
    batch_len = batch_size
    #if len(memory) < batch_size:
    #    #collect_data(game, model)
    #    batch_len=len(memory)

    batch = sample(memory, batch_len)
    # s1, a, r, s2 = batch
    # print(batch)
    s1 = np.array([i[0] for i in batch])
    a = np.array([i[1] for i in batch])
    r = np.array([i[2] for i in batch])
    s2 = np.array([i[3] for i in batch])
    terminal = np.array([i[4] for i in batch])
    q1 = get_q_values(s1)
    q2 = np.max(get_q_values(s2), axis=1)

    # is_discounted = np.array((r>0), dtype=np.int32).reshape(len(batch),)
    #print s1_Q.shape
    #print r.shape
    #print q2.shape
    #if(r > 0.5):
    q1[np.arange(q1.shape[0]), a] = r + (discount_factor * q2)  # * is_discounted  # Update q value with Bellman's equation if we have positive reward, otherwise just add reward
    #else:
    #    q1[np.arange(q1.shape[0]), a] = r + discount_factor * q2 - 100
    #for i in range(s1_Q.shape[0]):
    #    s1_Q[i]
    s1 = s1.reshape([s1.shape[0], -1, 1])
    model.fit(s1, q1, batch_size=batch_len, epochs=3, verbose=0, callbacks=[tensorboard])
    return batch_size  # to keep track of how many memories we have trained upon


def test_model(game, model, epoch):
    test_phase_rewards = []
    #tqdm.write("Testing Model....")
    for episode in range(1):
        game.new_episode()
        # tqdm.write("Episode {0}....".format(episode+1))
        reward = 123
        call = 1
        print("====================")
        while not game.is_episode_finished():
            print("call %s" % call)
            state = game.get_state()
            index = get_action(state)[0]  # get the best out of the prediction
            action = actions[index]
            
            #print(action)
            #print(model.predict(state.reshape([1,15,1]), batch_size=1, verbose=0))
            reward = game.make_action(action, epoch, print_output = True)
            call+=1
        print("====================\n")
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
parser.add_argument("-power", help="run power algorithm", action="store_true")
args = parser.parse_args(args=None, namespace=None)

run_power = False
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
if args.power:
    run_power = True
# add support for other arguments later. Like config file attributes and neural network parameters

''' END Arument Parsing'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to disable tensorflow debugging. It gets rather annoying
game = initialize_game()
global num_actions
num_actions = game.get_num_actions()
global actions
actions = np.array([list(a) for a in it.product([0, 1], repeat=num_actions)])
#actions = np.array([ [1,1,1,1,1,1,1,1], [1,1,1,1,1,1,1,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0], [1,0,1,0,1,0,1,1], [1,0,1,0,1,0,1,0] ])
#actions = sample(actions, 63)
#actions.append([1,1,1,1,1,1,1,1])
model = initialze_model(len(actions), (num_actions*2,1))
model.summary()
#tensorboard = keras.callbacks.TensorBoard(log_dir='Graph', histogram_freq=0,  
#          write_graph=True, write_images=True)
#tensorboard.set_model(model)



current_episode = 0

global_rewards = []
if(not run_power):
    for epoch in range(epochs):
        #print("\n=============================\nEpoch {0}....\n=============================\n".format(epoch + 1))
        print("Episode: %d" % (current_episode))
    
        

        
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
        rewards = np.array(test_model(game, model, epoch))
        global_rewards.append(rewards.mean())
        print("Mean reward for trial {0}, Total Mean reward: {1}\n".format(rewards.mean(), np.array(global_rewards).mean()))


        

    plt.plot(range(len(global_rewards)), global_rewards)
    plt.show()
    #final_train(game, model, memory)
    game.new_episode()
    state = game.get_state()
    a = get_action(state)
    print(actions[a])
    pass
else:
    from power import PoWER
    theta, eps, final_rewards, errors = PoWER(12, 256)
    print("theta: %s eps: %s final_rewards: %s errors: %s " % (theta, eps, final_rewards, errors))