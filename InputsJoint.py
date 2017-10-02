import zachopy.Talker
Talker = zachopy.Talker.Talker
import os
import numpy as np
from shutil import copyfile
from datetime import datetime
from astropy.table import Table

#  an object that reads in an input.init file, stores all of the information, and also makes a copy of that file to the directory everything will get saved to
class InputsJoint(Talker):
    '''InputsJoint object reads input.init information from all subdirectories and copies those files to the working detrender directry (if it's not already there).'''
    def __init__(self, jointdirectories, *directoryname):
        '''Initialize an Inputs object.'''

        Talker.__init__(self)

        self.jointdirectories = jointdirectories
        self.nightname = np.zeros(len(self.jointdirectories), dtype='S13')
        self.starlist = np.zeros(len(self.jointdirectories), dtype='S19')

        for n, subdir in enumerate(self.jointdirectories):
            self.n = n
            self.subdir = subdir
            if directoryname:
                self.speak('going into directory {0}'.format(directoryname[0]))
                self.directoryname = directoryname[0]
                self.readInputs()
            else: 
                self.createDirectory()
                self.readInputs()
        self.speak('successfully read in and joined input parameters')

    def createDirectory(self):
        ''' Create a new directory to put detrender stuff in'''

        self.speak('reading input file from {0}'.format(self.subdir))

        file = open(self.subdir+'/input.init')
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

        self.nightname[self.n] = dictionary['nightname']
        self.starlist[self.n] = dictionary['starlist']

        if self.n == 0:
            self.speak('creating new detrender directory')
            self.speak('where possible, input values from {0} will be used'.format(self.subdir))
            self.filename = dictionary['filename']


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
        
        self.speak('copying {0} and {1} to directory {2}'.format(self.subdir+'/input.init', self.subdir+'/'+self.starlist[self.n], self.directoryname))
        copyfile(self.subdir+'/input.init', self.directoryname+self.nightname[self.n]+'_input.init')
        copyfile(self.subdir+'/'+self.starlist[self.n], self.directoryname+self.nightname[self.n]+'_'+self.starlist[self.n])
            

    def readInputs(self):

        inputfilenames = []

        for file in os.listdir(self.directoryname):
            if file.endswith('_input.init'):
                inputfilenames.append(file)

        # be careful here! if for some reason the sorted list of your directories does not match up with the sorted list of the observation nights, you will introduce an error
        inputfilenames = sorted(inputfilenames)#, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))

        self.speak('reading {0} file from {1}'.format(inputfilenames[self.n], self.directoryname))

        file = open(self.directoryname+inputfilenames[self.n])
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

        if self.n == 0:
            self.filename = dictionary['filename']
            self.saveas = self.directoryname+'joint'

        self.nightname[self.n] = dictionary['nightname']

        if self.n == 0:
            self.target = []
            self.targetpx = []
            self.comparison = []
            self.comparisonpx = []

        self.starlist[self.n] = dictionary['starlist']
        star = Table.read(self.directoryname+self.nightname[self.n]+'_'+self.starlist[self.n], format='ascii', delimiter='&')
        comp, comppx = [], []
        for s in star:
            if s['star'] == 'targ': 
                self.target.append(s['aperture'])
                self.targetpx.append(s['extraction_window'])
            elif s['star'] == 'comp':
                comp.append(s['aperture'])
                comppx.append(s['extraction_window'])
        self.comparison.append(comp)
        self.comparisonpx.append(comppx)

        if self.n == 0:
            self.fitlabels = [dictionary['fitlabels']]
            self.T0 = float(dictionary['T0'])
            self.P = float(dictionary['P'])
            self.Tdur = float(dictionary['Tdur'])
            self.b = float(dictionary['b'])
            self.a = float(dictionary['a'])
            self.ecc = float(dictionary['ecc'])
            self.epochnum = [int(dictionary['epochnum'])]
            self.toff = [self.T0 + self.P*self.epochnum[self.n]]
            self.t0 = [str_to_bool(dictionary['t0'])]
            if type(self.t0) == bool: self.t0 = [self.toff]

        else:
            self.fitlabels.append(dictionary['fitlabels'])
            self.epochnum.append(int(dictionary['epochnum']))
            self.toff.append(self.T0 + self.P*self.epochnum[self.n])
            self.t0.append(str_to_bool(dictionary['t0']))
            if type(self.t0[-1]) == bool: self.t0[-1] = [self.toff]

        if self.n == 0:
            self.tranlabels = [dictionary['tranlabels']]
            self.tranparams = [[str_to_bool(i) for i in dictionary['tranparams']]]
            try: self.fixedrp = str_to_bool(dictionary['fixedrp'])
            except(TypeError): self.fixedrp = [str_to_bool(i) for i in dictionary['fixedrp']]
            self.tranbounds = [[[str_to_bool(i) for i in dictionary['tranbounds_low']], [str_to_bool(i) for i in dictionary['tranbounds_high']]]]
            self.wavelength_lims = [float(i) for i in dictionary['wavelength_lims']]
            if len(self.tranbounds[self.n][0]) != len(self.tranlabels[self.n]) or len(self.tranbounds[self.n][1]) != len(self.tranlabels[self.n]):
                self.speak('oh no! incorrect number of bounds!')
                return

            self.fitparams = [[1 for f in self.fitlabels[self.n]]]

            self.freeparambounds = [[], []]
            self.freeparamnames = []
            self.freeparamvalues = []
            for f, flabel in enumerate(self.fitlabels[self.n]):
                self.freeparambounds[0].append(True)
                self.freeparambounds[1].append(True)
                self.freeparamnames.append(flabel+str(self.n))
                self.freeparamvalues.append(self.fitparams[self.n][f])
            for t, tlabel in enumerate(self.tranlabels[self.n]):
                if type(self.tranbounds[self.n][0][t]) == bool and self.tranbounds[self.n][0][t] == False: continue
                if self.tranbounds[self.n][0][t] == 'Joint': continue
                self.freeparambounds[0].append(self.tranbounds[self.n][0][t])
                self.freeparambounds[1].append(self.tranbounds[self.n][1][t])
                self.freeparamnames.append(tlabel+str(self.n))
                self.freeparamvalues.append(self.tranparams[self.n][t])

        else:
            self.tranlabels.append(dictionary['tranlabels'])
            self.tranparams.append([str_to_bool(i) for i in dictionary['tranparams']])
            self.tranbounds.append([[str_to_bool(i) for i in dictionary['tranbounds_low']], [str_to_bool(i) for i in dictionary['tranbounds_high']]])

            self.fitparams.append([1 for f in self.fitlabels[self.n]])

            for f, flabel in enumerate(self.fitlabels[self.n]):
                self.freeparambounds[0].append(True)
                self.freeparambounds[1].append(True)
                self.freeparamnames.append(flabel+str(self.n))
                self.freeparamvalues.append(self.fitparams[self.n][f])
            for t, tlabel in enumerate(self.tranlabels[self.n]):
                if type(self.tranbounds[self.n][0][t]) == bool and self.tranbounds[self.n][0][t] == False: continue
                if self.tranbounds[self.n][0][t] == 'Joint': continue
                self.freeparambounds[0].append(self.tranbounds[self.n][0][t])
                self.freeparambounds[1].append(self.tranbounds[self.n][1][t])
                self.freeparamnames.append(tlabel+str(self.n))
                self.freeparamvalues.append(self.tranparams[self.n][t])
        
        if self.n == 0:
            self.binlen = str_to_bool(dictionary['binlen'])
            self.sigclip = float(dictionary['sigclip'])

            self.mcmccode = dictionary['mcmccode']
            self.nwalkers = int(dictionary['nwalkers'])
            self.nsteps = int(dictionary['nsteps'])
            self.burnin = int(dictionary['burnin'])

            self.optext = str_to_bool(dictionary['optext'])
            self.istarget = str_to_bool(dictionary['istarget'])
            self.isasymm = str_to_bool(dictionary['isasymm'])
            self.invvar = str_to_bool(dictionary['invvar'])
            self.ldmodel = str_to_bool(dictionary['ldmodel'])
            self.domcmc = str_to_bool(dictionary['domcmc'])

            self.mastern = [dictionary['mastern']]
            self.starmasterstr = [dictionary['starmasterstr']+self.mastern[self.n]+'.npy']

            self.datacubepath = [self.subdir+'/trimmed_cube.npy']
        else:
            self.mastern.append(dictionary['mastern'])
            self.starmasterstr.append(dictionary['starmasterstr']+self.mastern[self.n]+'.npy')

            self.datacubepath.append(self.subdir+'/trimmed_cube.npy')


                    
