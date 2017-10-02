import zachopy.Talker
import zachopy.Writer
Talker = zachopy.Talker.Talker
Writer = zachopy.Writer.Writer
from Inputs import Inputs
from InputsJoint import InputsJoint
from CubeReader import CubeReader
from CubeReaderJoint import CubeReaderJoint
from LCMaker import LCMaker
from LCMakerJoint import LCMakerJoint
from LMFitter import LMFitter
from LMFitterJoint import LMFitterJoint
from MCFitter import MCFitter
from MCFitterJoint import MCFitterJoint
import numpy as np
import os
from datetime import datetime

class Detrender(Talker, Writer):
    '''Detrenders are objects for detrending data output by mosasaurus.'''

    def __init__(self, joint, *directoryname):
        '''initialize from an input.init file'''

        # decide whether or not this Reducer is chatty
        Talker.__init__(self)
        
        self.joint = joint

        if directoryname: 
            self.speak('the detrender is starting in the folder {0}'.format(directoryname[0]))
            self.directoryname = str(directoryname[0])
        else: self.speak('making a new detrender')

        # setup all the components of the detrender
        if self.joint: 
            self.setup_joint()
        else: 
            self.setup()

        self.speak('detrender is ready to detrend')
    
    def setup(self):

        # load in the input parameters from input.init
        
        try: 
            self.inputs = Inputs(self.directoryname)
        except(AttributeError): 
            self.inputs = Inputs()
            self.directoryname = self.inputs.directoryname

        self.datacubepath = './trimmed_cube.npy'

        # try first loading in a saved minicube; if it doesn't exist read in the whole original cube
        self.cube = CubeReader(self)

        # try first to look for the files with the lcs already in them, eg. '7000.npy'
        self.lcs = LCMaker(self)

    def setup_joint(self):

        # load in the input parameters from multiple nights of data from input.init in each night's folder

        self.jointdirectories = [d for d in os.listdir('.') if os.path.isdir(os.path.join('.', d))]
        if 'run' in self.jointdirectories: self.jointdirectories.remove('run')
        self.jointdirectories = sorted(self.jointdirectories, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))

        try: 
            self.inputs = InputsJoint(self.jointdirectories, self.directoryname)
        except(AttributeError): 
            self.inputs = InputsJoint(self.jointdirectories)
            self.directoryname = self.inputs.directoryname

        # try first loading in a saved minicube; if it doesn't exist read in the whole original cube
        self.cube = CubeReaderJoint(self, self.jointdirectories)

        # try first to look for the files with the lcs already in them, eg. '7000.npy'
        self.lcs = LCMakerJoint(self, self.jointdirectories)
    
    def detrend(self):
        
        if self.joint: 

            self.speak('detrending data from nights {0} in directory {1}'.format(self.inputs.nightname, self.directoryname))

            for w, wavefile in enumerate(self.lcs.wavebin.wavefiles):
                if self.inputs.fixedrp != False:
                    rpind = np.where(np.array(self.inputs.tranlabels) == 'rp')[0][0]
                    self.inputs.tranparams[rpind] = self.inputs.fixedrp[w]
                    Writer.__init__(self, self.inputs.saveas+'_'+wavefile+'.txt')
                    self.speak('lmfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                    self.write('lmfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                self.lmfit = LMFitterJoint(self, wavefile)

            if self.inputs.domcmc:
                for wavefile in self.lcs.wavebin.wavefiles:
                    if self.inputs.fixedrp != False:
                        rpind = np.where(np.array(self.inputs.tranlabels) == 'rp')[0][0]
                        self.inputs.tranparams[rpind] = self.inputs.fixedrp[w]
                        Writer.__init__(self, self.inputs.saveas+'_'+wavefile+'.txt')
                        self.speak('mcfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                        self.write('mcfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                    self.mcfit = MCFitterJoint(self, wavefile)

        else:

            self.speak('detrending data from night {0} in directory {1}'.format(self.inputs.nightname, self.directoryname))

            for w, wavefile in enumerate(self.lcs.wavebin.wavefiles):
                if self.inputs.fixedrp != False:
                    rpind = np.where(np.array(self.inputs.tranlabels) == 'rp')[0][0]
                    self.inputs.tranparams[rpind] = self.inputs.fixedrp[w]
                    Writer.__init__(self, self.inputs.saveas+'_'+wavefile+'.txt')
                    self.speak('lmfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                    self.write('lmfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                self.lmfit = LMFitter(self, wavefile)

            if self.inputs.domcmc:
                for wavefile in self.lcs.wavebin.wavefiles:
                    if self.inputs.fixedrp != False:
                        rpind = np.where(np.array(self.inputs.tranlabels) == 'rp')[0][0]
                        self.inputs.tranparams[rpind] = self.inputs.fixedrp[w]
                        Writer.__init__(self, self.inputs.saveas+'_'+wavefile+'.txt')
                        self.speak('mcfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                        self.write('mcfit will used a fixed rp/rs value of {0}'.format(self.inputs.tranparams[rpind]))
                    self.mcfit = MCFitter(self, wavefile)

        self.speak('detrender done')
        self.speak('detrendersaurus has done all it can do for your data. goodbye.')
