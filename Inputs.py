import zachopy.Talker
Talker = zachopy.Talker.Talker
import os
import numpy as np
from shutil import copyfile
from datetime import datetime
from astropy.table import Table
import string

#  an object that reads in an input.init file, stores all of the information, and also makes a copy of that file to the directory everything will get saved to
class Inputs(Talker):
    '''Inputs object reads input.init information and copies that file to the working directry (if it's not already there).'''
    def __init__(self, *directoryname):
        '''Initialize an Inputs object.'''

        Talker.__init__(self)


        if directoryname:
            self.speak('going into directory {0}'.format(directoryname[0]))
            self.directoryname = directoryname[0]
            self.readInputs()
        else: 
            self.createDirectory()
            self.readInputs()

    def createDirectory(self):
        ''' Create a new directory to put detrender stuff in'''

        self.speak('creating new detrender directory'.format())

        file = open('input.init')
        lines = file.readlines()
        dictionary = {}
        for i in range(3):
          if lines[i] != '\n' and lines[i][0] != '#':
            split = lines[i].split()
            key = split[0]
            entries = split[1:]
            if len(entries) == 1:
              entries = entries[0]
            dictionary[key] = entries

        self.filename = str(dictionary['filename'])
        self.nightname = str(dictionary['nightname'])
        self.starlist = str(dictionary['starlist'])

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

        self.speak('copying {0} and {1} to directory {2}'.format('input.init', self.starlist, self.directoryname))
        copyfile('input.init', self.directoryname+self.nightname+'_input.init')
        copyfile(self.starlist, self.directoryname+self.nightname+'_'+self.starlist)
            

    def readInputs(self):

        for file in os.listdir(self.directoryname):
            if file.endswith('_input.init'):
                inputfilename = file

        self.speak('reading {0} file from {1}'.format(inputfilename, self.directoryname))

        file = open(self.directoryname+inputfilename)
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
                except(ValueError): 
                    if s in dictionary.keys(): return float(dictionary[s])
                    else: return str(s)


        self.filename = dictionary['filename']
        self.nightname = dictionary['nightname']
        self.saveas = self.directoryname+self.nightname


        self.starlist = dictionary['starlist']
        star = Table.read(self.directoryname+self.nightname+'_'+self.starlist, format='ascii', delimiter='&')
        self.comparison = []
        self.comparisonpx = []
        for s in star:
            if s['star'] == 'targ': 
                self.target = s['aperture']
                self.targetpx = s['extraction_window']
            elif s['star'] == 'comp':
                self.comparison.append(s['aperture'])
                self.comparisonpx.append(s['extraction_window'])

        self.optext = str_to_bool(dictionary['optext'])
        self.istarget = str_to_bool(dictionary['istarget'])
        self.isasymm = str_to_bool(dictionary['isasymm'])
        self.invvar = str_to_bool(dictionary['invvar'])
        self.ldmodel = str_to_bool(dictionary['ldmodel'])
        self.domcmc = str_to_bool(dictionary['domcmc'])

        self.fitlabels = dictionary['fitlabels']
        if type(self.fitlabels) == str: self.fitlabels = [self.fitlabels]
        self.polyfit = int(dictionary['polyfit'])
        self.polylabels = [string.uppercase[x] for x in range(self.polyfit)]
        self.T0 = float(dictionary['T0'])
        self.P = float(dictionary['P'])
        self.Tdur = float(dictionary['Tdur'])
        self.inc = float(dictionary['inc'])
        self.a = float(dictionary['a'])
        self.ecc = float(dictionary['ecc'])
        self.epochnum = int(dictionary['epochnum'])
        self.toff = self.T0 + self.P*self.epochnum


        self.tranlabels = dictionary['tranlabels']
        self.tranparams = [str_to_bool(i) for i in dictionary['tranparams']]
        try: self.fixedrp = str_to_bool(dictionary['fixedrp'])
        except(TypeError): self.fixedrp = [str_to_bool(i) for i in dictionary['fixedrp']]
        self.tranbounds = [[str_to_bool(i) for i in dictionary['tranbounds_low']], [str_to_bool(i) for i in dictionary['tranbounds_high']]]
        self.wavelength_lims = [float(i) for i in dictionary['wavelength_lims']]
        if len(self.tranbounds[0]) != len(self.tranlabels) or len(self.tranbounds[1]) != len(self.tranlabels):
            self.speak('oh no! incorrect number of bounds!')
            return

        self.fitparams = [1 for f in self.fitlabels]
        self.polyparams = [1 for p in self.polylabels]

        self.freeparambounds = [[], []]
        self.freeparamnames = []
        self.freeparamvalues = []
        for p, plabel in enumerate(self.polylabels):
            self.freeparambounds[0].append(True)
            self.freeparambounds[1].append(True)
            self.freeparamnames.append(plabel)
            self.freeparamvalues.append(self.polyparams[p])
        for f, flabel in enumerate(self.fitlabels):
            self.freeparambounds[0].append(True)
            self.freeparambounds[1].append(True)
            self.freeparamnames.append(flabel)
            self.freeparamvalues.append(self.fitparams[f])
        for t, tlabel in enumerate(self.tranlabels):
            if type(self.tranbounds[0][t]) == bool and self.tranbounds[0][t] == False: continue
            if self.tranbounds[0][t] == 'Joint': continue
            self.freeparambounds[0].append(self.tranbounds[0][t])
            self.freeparambounds[1].append(self.tranbounds[1][t])
            self.freeparamnames.append(tlabel)
            self.freeparamvalues.append(self.tranparams[t])

        dtind = int(np.where(np.array(self.tranlabels) == 'dt')[0])
        self.t0 = self.toff + self.tranparams[dtind]

        self.binlen = str_to_bool(dictionary['binlen'])
        self.sigclip = float(dictionary['sigclip'])

        self.mcmccode = dictionary['mcmccode']
        if self.mcmccode == 'dynesty': pass
        elif self.mcmccode == 'emcee':
            self.nwalkers = int(dictionary['nwalkers'])
            self.nsteps = int(dictionary['nsteps'])
            self.burnin = int(dictionary['burnin'])

        self.mastern = dictionary['mastern']
        self.starmasterstr = dictionary['starmasterstr']+self.mastern+'.npy'

        self.speak('successfully read in input parameters')
                    

