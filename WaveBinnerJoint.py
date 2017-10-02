import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np

class WaveBinnerJoint(Talker):

    '''this class will bin the data into even wavelength bins'''

    def __init__(self, detrender, jointdirectories):

        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.cube = self.detrender.cube
        self.jointdirectories = jointdirectories

        self.binindices = []

        for n, subdir in enumerate(self.jointdirectories):
            self.n = n
            self.subdir = subdir
            if 'wavebin' in self.cube.subcube[self.n].keys():
                self.speak('reading in wavebin parameters from subcube saved in {0}'.format(self.detrender.directoryname))
                self.binindices.append(self.cube.subcube[self.n]['wavebin']['binindices'])
                if self.n == 0:
                    self.wavelims = self.cube.subcube[self.n]['wavebin']['wavelims']
                    self.wavefiles = self.cube.subcube[self.n]['wavebin']['wavefiles']
            else: 
                self.makeBinIndices()
                self.speak('saving wavebin properties to the subcube')
                np.save(self.inputs.saveas+'_subcube.npy', self.cube.subcube)


    def makeBinIndices(self):

        self.speak('creating binindices array for {0} which will help to make wavelength binned lighcurves'.format(self.subdir))

        numexps, numwave = self.cube.subcube[self.n]['wavelengths'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]].shape

        if self.n == 0:
            waverange = self.inputs.wavelength_lims[1] - self.inputs.wavelength_lims[0]
            if self.inputs.binlen == 'all': self.binlen = waverange
            else: self.binlen = self.inputs.binlen
            self.numbins = int(np.floor(waverange/self.binlen))
            self.binlen = waverange/float(self.numbins)
            self.wavelims = []
            [self.wavelims.append((self.inputs.wavelength_lims[0]+(i*self.binlen), self.inputs.wavelength_lims[0]+((i+1)*self.binlen))) for i in range(int(self.numbins))]
        binindices = np.zeros((numexps, int(len(self.inputs.comparison[self.n])+1), self.numbins, numwave))

        starmaster = np.load(self.inputs.starmasterstr[self.n])[()]

        #self.binnedcube_targ = np.zeros((numexps, self.numbins, numwave))
        for n in range(numexps):
            for i, wavelim in enumerate(self.wavelims):
                  
                minwave_interp1 = np.interp(wavelim[0], self.cube.subcube[self.n]['wavelengths'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]][n], starmaster['wavelength'])
                minwave_interp = np.interp(minwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                maxwave_interp1 = np.interp(wavelim[1], self.cube.subcube[self.n]['wavelengths'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]][n], starmaster['wavelength'])
                maxwave_interp = np.interp(maxwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                minwaveind = int(np.ceil(minwave_interp))
                minwaveextra = minwaveind - minwave_interp
                maxwaveind = int(np.floor(maxwave_interp))
                maxwaveextra = maxwave_interp - maxwaveind
                indarray = np.zeros((numwave))
                indarray[minwaveind:maxwaveind] = 1
                indarray[minwaveind-1] = minwaveextra
                indarray[maxwaveind] = maxwaveextra

                binindices[n][0][i] = indarray

        for n in range(numexps):
            for s in range(len(self.inputs.comparison[self.n])):
                for i, wavelim in enumerate(self.wavelims):
                      
                    minwave_interp1 = np.interp(wavelim[0], self.cube.subcube[self.n]['wavelengths'][self.inputs.comparison[self.n][s]][self.inputs.comparisonpx[self.n][s]][n], starmaster['wavelength'])
                    minwave_interp = np.interp(minwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                    maxwave_interp1 = np.interp(wavelim[1], self.cube.subcube[self.n]['wavelengths'][self.inputs.comparison[self.n][s]][self.inputs.comparisonpx[self.n][s]][n], starmaster['wavelength'])
                    maxwave_interp = np.interp(maxwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                    minwaveind = int(np.ceil(minwave_interp))
                    minwaveextra = minwaveind - minwave_interp
                    maxwaveind = int(np.floor(maxwave_interp))
                    maxwaveextra = maxwave_interp - maxwaveind
                    indarray = np.zeros((numwave))
                    indarray[minwaveind:maxwaveind] = 1
                    indarray[minwaveind-1] = minwaveextra
                    indarray[maxwaveind] = maxwaveextra

                    binindices[n][s+1][i] = indarray

        self.binindices.append(binindices)

        self.cube.subcube[self.n]['wavebin'] = {}
        self.cube.subcube[self.n]['wavebin']['binindices'] = binindices
        self.cube.subcube[self.n]['wavebin']['wavelims'] = self.wavelims
        if self.n == 0: self.wavefiles = [str(i[0])+'-'+str(i[1]) for i in self.wavelims]
        self.cube.subcube[self.n]['wavebin']['wavefiles'] = self.wavefiles


