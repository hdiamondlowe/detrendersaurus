'''If something inherits from Writer, then we can write text to an output txt file in a relatively standard way.'''
import textwrap, numpy as np

class Writer(object):
    '''Objects that inherit from Writer have a write('hello!') method that writes to an outpit file that you must specify at instatiation.'''
    def __init__(self, outputfile):
        self.outputfile = outputfile

    def write(self, string=''):
        f = open(self.outputfile, 'a')
        f.write(string+'\n')
        f.close()
