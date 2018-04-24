
from deepmusic.moduleloader import ModuleLoader
# predicts next key
from deepmusic.keybordcell import KeybordCell
# encapsulate song data so we can run get_scale, get_relative_methods
import deepmusic.songstruct as music
import numpy as np
import os
import tensorflow as tf


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\GenerateMusic')


def build_network(self):
    # create computation graph, encapsulate session and the graph init
    input_dim = ModuleLoader.batch_builders.get_module().get_input_dim()

    # note data
    with tf.name_scope('placeholder_inputs'):
        self.inputs = [
            tf.placeholder(tf.float32, # numerical data
                           [self.batch_size, input_dim], # how much data
                           name = 'input'
                           )
        ]

    # target 88 key, binary classification problem
    with tf.name_scope('placeholder_targets')
        self.targets = [
            tf.placeholder(tf.int32, # 0 / 1
                           [self.batch_size],
                           name = 'target'
                           )
        ]

    # previous hidden state
    with tf.name_scope('placeholder_use_prev')
        self.use_prev = [
            tf.placeholder(tf.bool,
                           [],
                           name = 'use_prev'
                           )
        ]

    # define our network     



def main():



    return 0


if __name__ == '__main__':
    main()
