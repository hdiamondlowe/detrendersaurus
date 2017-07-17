import zachopy.Talker
Talker = zachopy.Talker.Talker
import os
from shutil import copyfile
from datetime import datetime
from astropy.table import Table

#  an object that reads in an input.init file, stores all of the information, and also makes a copy of that file to the directory everything will get saved to
class Inputs(Talker):
    '''Inputs object reads input.init information and copies that file to the working directry (if it's not already there).'''
    def __init__(self, detrender, *directoryname):
        '''Initialize an Inputs object.'''

        Talker.__init__(self)

        self.detrender = detrender

        if directoryname:
            self.speak('going into directory {0}'.format(directoryname[0]))
            self.directoryname = directoryname[0]
            self.readInputs()
        else: self.createDirectory()

    def createDirectory(self):
        ''' Create a new directory to put detrender stuff in'''

        self.speak('creating new detrender directory'.format())

        file = open('input.init')
        lines = file.readlines()
        dictionary = {}
        for i in range(2):
          if lines[i] != '\n' and lines[i][0] != '#':
            split = lines[i].split()
            key = split[0]
            entries = split[1:]
            if len(entries) == 1:
              entries = entries[0]
            dictionary[key] = entries

        self.filename = str(dictionary['filename'])
        self.nightname = str(dictionary['nightname'])

        # create working folder for the files
        dt = datetime.now()
        runpath = dt.strftime('%Y-%m-%d-%H:%M_')+self.filename+'/'
        directorypath = 'run' + '/'
        if not os.path.exists(directorypath):
            os.makedirs(directorypath)
        if os.path.exists(directorypath+runpath):
            self.speak('run path already exists! you are writing over some files...')
        else:
            os.makedirs(directorypath+runpath)

        self.directoryname = directorypath+runpath
        self.saveas = self.directoryname+self.nightname
        self.outfile = self.saveas+'_output_'

        self.speak('copying "input.init" to the new directory')
        copyfile('./input.init', self.directoryname+self.nightname+'_input.init')

        self.readInputs()

    def readInputs(self):

        for file in os.listdir(self.directoryname):
            if file.endswith('input.init'):
                inputfile = file

        self.speak('trying to read {0} file from {1}'.format(inputfile, self.directoryname))

        file = open(self.directoryname+inputfile)
        lines = file.readlines()
        dictionary = {}
        for i in range(len(lines)):
          if lines[i] != '\n' and lines[i][0] != '#':
            split = lines[i].split()
            key = split[0]
            entries = split[1:]
            if len(entries) == 1:
              entries = entries[0]
            dictionary[key] = entries

        def str_to_bool(s):
            if s == 'True':
                return True
            elif s == 'False':
                return False
            else:
                try: return float(s)
                except(ValueError): return str(dictionary[s])


        self.filename = dictionary['filename']
        self.nightname = dictionary['nightname']
        self.saveas = self.directoryname+self.nightname
        self.outfile = self.saveas+'_output_'

        try: os.rename(self.directoryname+'input.init', self.saveas+'_input.init')
        except(OSError): pass

        self.starlist = dictionary['starlist']
        star = Table.read(self.starlist, format='ascii', delimiter='&')
        self.comparison = []
        self.comparisonpx = []
        for s in star:
            if s['star'] == 'targ': 
                self.target = s['aperture']
                self.targetpx = s['extraction_window']
            elif s['star'] == 'comp':
                self.comparison.append(s['aperture'])
                self.comparisonpx.append(s['extraction_window'])

        self.fitlabels = dictionary['fitlabels']
        self.T0 = float(dictionary['T0'])
        self.P = float(dictionary['P'])
        self.Tdur = float(dictionary['Tdur'])
        self.b = float(dictionary['b'])
        self.a = float(dictionary['a'])
        self.ecc = float(dictionary['ecc'])
        self.epochnum = int(dictionary['epochnum'])
        self.toff = self.T0 + self.P*self.epochnum
        self.t0 = str_to_bool(dictionary['t0'])
        if type(self.t0) == bool: self.t0 = self.toff

        self.tranlabels = dictionary['tranlabels']
        self.tranparams = [str_to_bool(i) for i in dictionary['tranparams']]
        self.tranbounds = [[str_to_bool(i) for i in dictionary['tranbounds_low']], [str_to_bool(i) for i in dictionary['tranbounds_high']]]
        self.wavelength_lims = [float(i) for i in dictionary['wavelength_lims']]
        if len(self.tranbounds[0]) != len(self.tranlabels) or len(self.tranbounds[1]) != len(self.tranlabels):
            self.speak('oh no! incorrect number of bounds!')
            return

        self.parambounds = [[True for t in self.fitlabels],[True for t in self.fitlabels]]
        self.paramlabels, self.paramvalues = [f for f in self.fitlabels], [1 for f in self.fitlabels]
        for i in range(len(self.tranparams)):
            if type(self.tranbounds[0][i]) != bool or self.tranbounds[0][i] == True:
                self.paramlabels.append(self.tranlabels[i])
                self.paramvalues.append(self.tranparams[i])
                self.parambounds[0].append(self.tranbounds[0][i])
                self.parambounds[1].append(self.tranbounds[1][i])

        self.binlen = str_to_bool(dictionary['binlen'])
        self.sigclip = float(dictionary['sigclip'])
        self.nwalkers = int(dictionary['nwalkers'])
        self.nsteps = int(dictionary['nsteps'])
        self.burnin = int(dictionary['burnin'])

        self.optext = str_to_bool(dictionary['optext'])
        self.istarget = str_to_bool(dictionary['istarget'])
        self.isasymm = str_to_bool(dictionary['isasymm'])
        self.invvar = str_to_bool(dictionary['invvar'])
        self.ldmodel = str_to_bool(dictionary['ldmodel'])
        self.domcmc = str_to_bool(dictionary['domcmc'])
        self.saveparams = str_to_bool(dictionary['saveparams'])

        self.mastern = dictionary['mastern']
        self.starmasterstr = dictionary['starmasterstr']+self.mastern+'.npy'

        self.speak('successfully read in input parameters')
                    

