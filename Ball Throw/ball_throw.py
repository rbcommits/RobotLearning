import argparse
import itertools as it
import numpy as np

import tensorflow as tf
import os

import cPickle
import ConfigParser
import keras

from random import sample, random, randint
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from time import sleep
from tqdm import trange, tqdm


#neural network variables
learning_rate = 25e-5
epochs = 20
batch_size = 256                     # 256 seems optimal for my 2 gig GPU
memory_size = 10000                 # number of entries to store in it's memory. Seems rather low, could increase it as we proceed
discount_factor = 0.99
collection_per_epoch = 2000
testing_per_epoch = 10
save_model = True
save_interval = 10000                # number of training episodes after which the model is saved to the disk. I use a lowe number since my gPU is slow and I usually quit prematurely
load_model = True
epsilon = 1.0
observe_episodes = 1000
memory = []
model_savefile_location = "models/Jarvis-{0}-{1}x{2}-{3}".format(5, image_resolution[0], image_resolution[1], learning_rate)
model_state_shape = (1, image_resolution[0], image_resolution[1], channels)
collect_data_bool = False
train_data_bool = False
test_data_bool = False
train_and_collect = False
# other functional variables
jarvis_config="jarvis.cfg"

def initialize_vizdoom():
    print "Initializing Robot Engine....."
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
    return game

def initialze_model(num_output, num_input):
    print "Initializing JARVIS...."
    ''' trying to follow the deep mind structure.
        - Layer 1: Dense,             Activation = Relu,   output = 32, strides = 4, Kernal size 8x8
        - Layer 2: Dense,             Activation = Relu,   output = 64, strides = 2, Kernal size 4x4
        - Layer 3: Dense,             Activation = Relu,   output = 64, strides = 1, Kernal size 3x3
        - Layer 5: Fully Conencted/Dense,   Activation = Relu,   output = 512,
        - Layer 6: Fully Conencted/Dense,   Activation = Linear, output = (1x1xnum_actions)
    '''

    model = Sequential()

    model.add(Dense(units=32, activation='relu', input_shape=num_input, bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Conv2D(units=64, activation='relu', bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Conv2D(units=128, activation='relu', bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dense(units=512, activation='relu', bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dense(units=num_output, activation='linear', bias_initializer=keras.initializers.Constant(0.1)))
    rms_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='mse',
              optimizer=rms_optimizer, metrics=['mse'])



    if load_model:
        print "Loading weights...."
        model.load_weights(model_savefile_location, by_name=True)
    print "Initialization finished...."
    print "outputs ", num_output
    return model

def get_action(batch):
    result = (model.predict(batch, batch_size=batch.shape[0], verbose=0))
    return np.argmax(result,1)

def get_q_values(batch):
    return (model.predict(batch, batch_size=batch.shape[0], verbose=0))

def get_reward():
    # simulate ball throw here
    reward = 0
    return reward


def collect_data(game, model, epoch):
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

    index = 0
    state1 = game.get_state()
    if random() <= exploration_rate(epoch): #or training_episodes_finished < observe_episodes:
        index = randint(0, num_actions - 1) # make a completly random action
    else:
        index = get_action(state1)[0]#np.argmax(model.predict(image, batch_size=1),1) # get the best out of the prediction
    reward = game.make_action(actions[index], frame_repeat)
    state2 = game.get_state()
        memory.append([state1, index, reward, state2]) # here we use the image to describe state. we can add other parameters later as well!
    if len(memory) > memory_size:
        memory.pop(0) # if we exceed, remove old memories


def train_model(game, model, memory):
    if (len(memory) < batch_size) and not train_data_bool:
        return 0  #to only train when we have enough data to train on.
    batch = sample(memory, batch_size)
    #s1, a, r, s2 = batch
    #print(batch)
    s1 = np.array([i[0] for i in batch])
    a = np.array([i[1] for i in batch])
    r = np.array([i[2] for i in batch])
    s2 = np.array([i[3] for i in batch])
    s1_Q = get_q_values(s1)
    q2 = np.max(get_q_values(s2), axis=1)
    #is_discounted = np.array((r>0), dtype=np.int32).reshape(len(batch),)
    s1_Q[np.arange(s1_Q.shape[0]), a] = r + discount_factor * q2 #* is_discounted  # Update q value with Bellman's equation if we have positive reward, otherwise just add reward
    model.fit(s1, s1_Q, batch_size=batch_size, epochs=1, verbose=0)
    return batch_size # to keep track of how many memories we have trained upon

def test_model(game, model):
    test_phase_rewards = []
    tqdm.write("Testing Model....")
    for episode in trange(testing_per_epoch, leave=False):
        game.new_episode()
        #tqdm.write("Episode {0}....".format(episode+1))
        while not game.is_episode_finished():
            state = game.get_state()
            index = get_action(state)[0] # get the best out of the prediction
            reward = game.make_action(actions[index])

        test_phase_rewards.append(game.get_total_reward())
    return test_phase_rewards

def init_ros_publisher(topic):
    global publisher
    print("Initializing ROS publisher node {0}".format(__name__))
    rospy.init_node(__name__)
    publisher = rospy.Publisher(topic, String, queue_size=10)
    print("ROS node initialized. Commencing data collection")

def read_from_config(cfg_file):
    global total_trained_images

    config = ConfigParser.RawConfigParser()
    config.read(cfg_file)
    total_trained_images = config.getint('Training', 'total_training_data')
    return config

def write_to_config(config, cfg_file):
    global total_trained_images

    config.set('Training', 'total_training_data', total_trained_images)
    with open(cfg_file, 'wb') as configfile:
        config.write(configfile)


''' BEGIN Argument Parsing '''
# Parse arguments for location of cfg file and wad file. Otherwise use defaults
parser = argparse.ArgumentParser()
parser.add_argument("-cfg", help="location of the config file to use for the map")
parser.add_argument("-wad", help="location of the wad file which contains the map")
parser.add_argument("-c", action="store_true", help="If you want to collect data for training")
parser.add_argument("-t", action="store_true", help="if you want to train on previously collected data")
parser.add_argument("-te", action="store_true", help="if you want to test your model playing")
parser.add_argument("-ros", action="store_true", help="if you want to use ROS during collection phase to collect from multiple instances")
parser.add_argument("-name", help="if you want to provide a custom name for the script. useful when launching multiple instances from a master")
parser.add_argument("-topic", help="name of the topic this ROS node will publish to")
parser.add_argument("-l", action="store_true", help="if you want to load a previously saved model")
parser.add_argument("-e", help="number of epochs")
parser.add_argument("-tac", action="store_true", help="if you want to perform training and collection at the same time")
args = parser.parse_args(args=None, namespace=None)

if args.cfg:
    config_file_path = args.cfg
if args.wad:
    wad_file_path = args.wad
if args.c:
    collect_data_bool = True
    print("Data collection enabled")
if args.t:
    train_data_bool = True
    print("Model training enabled")
if args.te:
    test_data_bool = True
    print("Model Testing enabled")
if args.name:
    __name__ = args.name
if args.ros:
    use_ROS = True
    topic = args.topic
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to disable tensorflow debugging. It gets rather annoying
game = initialize_vizdoom()
global num_actions
num_buttons = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=num_buttons)]       # Get all possible actions as one hot encoding [0 0 1], [0 1 0] [1 0 0] etc
num_actions = len(actions)
model = initialze_model(num_actions)
model.summary()
get_action(blank_image) # has no use except forcing TF/Theano to reinitialize earlier and let my outputs print properly
global trained_images
trained_images = 0

config_file = read_from_config(jarvis_config)
if use_ROS:
    init_ros_publisher(topic)

for epoch in range(epochs):
    print "\n=============================\nEpoch {0}....\n=============================\n".format(epoch + 1)
    current_episode = 0
    global_rewards = []

    if train_and_collect:
        tqdm.write("\tStarting learning episode")
        game.new_episode()
        for learning_episode in trange(collection_per_epoch, leave=False):
            collect_data(game, model, epoch)
            trained_images += train_model(game, model, memory)
            if game.is_episode_finished():
                game.new_episode()
            if current_episode % int(collection_per_epoch/2) == 0:
                tqdm.write("\tSaving model {0}".format(model_savefile_location))
                model.save_weights(model_savefile_location, overwrite=True)
                total_trained_images+=trained_images
                write_to_config(config_file, jarvis_config)
            current_episode +=1

    if collect_data_bool:
        for learning_episode in trange(collection_per_epoch, leave=False):
            tqdm.write("\tStarting collection episode {0}".format(current_episode))
            collect_data(game, model, epoch)
            current_episode+=1

    if train_data_bool:
        if train_from_file:
            files = os.listdir("data/balanced_data/")
            current_file = 0
            outer_loop = tqdm(files, desc="Training on file {0}".format(files[current_file]), leave = None)
            for data_file in outer_loop:
                outer_loop.set_description("Training on file {0}".format(files[current_file]))
                numpy_data = np.load("data/balanced_data/{0}".format(data_file))
                current_read = 0
                inner_loop = tqdm(xrange(0, numpy_data.shape[0], batch_size), leave = None, desc="Trained {} images from current file".format(current_read))
                for file_length in inner_loop:
                    trained_images += train_model(game, model, numpy_data)
                    current_read+=batch_size
                    inner_loop.set_description("Trained {} images from current file".format(current_read))
                tqdm.write("\tSaving model {0}".format(model_savefile_location))
                model.save_weights(model_savefile_location, overwrite=True)
                current_file += 1

        total_trained_images+=trained_images
        write_to_config(config_file, jarvis_config)
    if test_data_bool:
        total_rewards = np.array(test_model(game, model))
        print(type(total_rewards))
        print(total_rewards.shape)
        print(total_rewards)
        print "Collection, training, testing cycle completed. Printing Statistics: \nMax reward{0}, Min reward: {1}, Mean reward {2}\n".format(total_rewards.max(), total_rewards.min(), total_rewards.mean())
        print("Total Trained data: ", trained_images)
print("Doing Final Testing")
while True:
    total_rewards = np.array(test_model(game, model))
    print(type(total_rewards))
    print(total_rewards.shape)
    print(total_rewards)
    print "Collection, training, testing cycle completed. Printing Statistics: \nMax reward{0}, Min reward: {1}, Mean reward {2}\n".format(total_rewards.max(), total_rewards.min(), total_rewards.mean())
    print("Total Trained data: ", trained_images)
