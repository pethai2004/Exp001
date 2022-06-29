import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np

class LogValue:
    '''use in GPU cloud, alternative to Tensorboard'''
    def __init__(self):
        self.registers = {}
        self.register_name = []
        
    def add_name(self, name):
        self.registers.update({name: []})
        self.register_name.append(name)

    def add_value(self, name, value):

        #assert len(value.shape) == 1
        if name not in self.register_name:
            self.add_name(name)

        self.registers[name].append(value)

    def get_value(self, name):
        return np.squeeze(self.registers[name])
    
    def reset_value(self, name=None):
        if name is None:
            for n in self.register_name:
                self.registers[n] = []
        else:
            self.registers[name] = []