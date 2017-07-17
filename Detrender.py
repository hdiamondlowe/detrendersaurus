import zachopy.Talker
Talker = zachopy.Talker.Talker
from Inputs import Inputs
from CubeReader import CubeReader
from LCMaker import LCMaker

class Detrender(Talker):
    '''Detrenders are objects for detrending data output by mosasaurus.'''

    def __init__(self, *directoryname):
        '''initialize from an input.init file'''

        # decide whether or not this Reducer is chatty
        Talker.__init__(self)
        
        if directoryname: 
            self.speak('the detrender is starting in the folder {0}'.format(directoryname[0]))
            self.directoryname = str(directoryname[0])
        else: self.speak('making a new detrender')

        # setup all the components of the detrender
        self.setup()

        self.speak('detrender is ready to detrend')
    
    def setup(self):

        # load in the input parameters from input.init
        
        try: 
            self.inputs = Inputs(self, self.directoryname)
        except(AttributeError): 
            self.inputs = Inputs(self)
            self.directoryname = self.inputs.directoryname

        self.datacubepath = './trimmed_cube.npy'

        # try first loading in a saved minicube; if it doesn't exist read in the whole original cube
        self.cube = CubeReader(self)

        # try first to look for the files with the lcs already in them, eg. '7000.npy'
        self.lcs = LCMaker(self)
    
    def detrend(self):
        
        self.speak('detrending data from night {0} in directory {1}'.format(
                        self.inputs.nightname, self.directoryname))

        # set up the calibrations and mask
        self.calib.setup()
        self.mask.setup()

        # load the headers for this observation
        self.obs.loadHeaders()

        # create an observing log from all the image headers from this night
        self.night.obsLog()

        # loop over exposures and apertures
        self.mask.extractEverything(remake=remake)
