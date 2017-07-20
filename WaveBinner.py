import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np

class WaveBinner(Talker):

    '''this class will bin the data into even wavelength bins'''

    def __init__(self, detrender):

        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.cube = self.detrender.cube

        if 'wavebin' in self.cube.subcube.keys():
            self.speak('reading in wavebin parameters from subcube saved in {0}'.format(self.detrender.directoryname))
            self.binindices = self.cube.subcube['wavebin']['binindices']
            self.wavelims = self.cube.subcube['wavebin']['wavelims']
            self.wavefiles = self.cube.subcube['wavebin']['wavefiles']
        else: self.makeBinIndices()


    def makeBinIndices(self):

        self.speak('creating binindices array which will help to make wavelength binned lighcurves')

        numexps, numwave = self.cube.subcube['wavelengths'][self.inputs.target][self.inputs.targetpx].shape
        waverange = self.inputs.wavelength_lims[1] - self.inputs.wavelength_lims[0]

        if self.inputs.binlen == 'all': self.binlen = waverange
        else: self.binlen = self.inputs.binlen
        self.numbins = int(np.floor(waverange/self.binlen))
        self.binlen = waverange/float(self.numbins)
        self.wavelims = []
        [self.wavelims.append((self.inputs.wavelength_lims[0]+(i*self.binlen), self.inputs.wavelength_lims[0]+((i+1)*self.binlen))) for i in range(int(self.numbins))]
        self.binindices = np.zeros((numexps, int(len(self.inputs.comparison)+1), self.numbins, numwave))

        starmaster = np.load(self.inputs.starmasterstr)[()]

        #self.binnedcube_targ = np.zeros((numexps, self.numbins, numwave))
        for n in range(numexps):
            for i, wavelim in enumerate(self.wavelims):
                  
                minwave_interp1 = np.interp(wavelim[0], self.cube.subcube['wavelengths'][self.inputs.target][self.inputs.targetpx][n], starmaster['wavelength'])
                minwave_interp = np.interp(minwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                maxwave_interp1 = np.interp(wavelim[1], self.cube.subcube['wavelengths'][self.inputs.target][self.inputs.targetpx][n], starmaster['wavelength'])
                maxwave_interp = np.interp(maxwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                minwaveind = int(np.ceil(minwave_interp))
                minwaveextra = minwaveind - minwave_interp
                maxwaveind = int(np.floor(maxwave_interp))
                maxwaveextra = maxwave_interp - maxwaveind
                indarray = np.zeros((numwave))
                indarray[minwaveind:maxwaveind] = 1
                indarray[minwaveind-1] = minwaveextra
                indarray[maxwaveind] = maxwaveextra

                self.binindices[n][0][i] = indarray

        for n in range(numexps):
            for s in range(len(self.inputs.comparison)):
                for i, wavelim in enumerate(self.wavelims):
                      
                    minwave_interp1 = np.interp(wavelim[0], self.cube.subcube['wavelengths'][self.inputs.comparison[s]][self.inputs.comparisonpx[s]][n], starmaster['wavelength'])
                    minwave_interp = np.interp(minwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                    maxwave_interp1 = np.interp(wavelim[1], self.cube.subcube['wavelengths'][self.inputs.comparison[s]][self.inputs.comparisonpx[s]][n], starmaster['wavelength'])
                    maxwave_interp = np.interp(maxwave_interp1, starmaster['wavelength'], starmaster['w']) - starmaster['w'][0]
                    minwaveind = int(np.ceil(minwave_interp))
                    minwaveextra = minwaveind - minwave_interp
                    maxwaveind = int(np.floor(maxwave_interp))
                    maxwaveextra = maxwave_interp - maxwaveind
                    indarray = np.zeros((numwave))
                    indarray[minwaveind:maxwaveind] = 1
                    indarray[minwaveind-1] = minwaveextra
                    indarray[maxwaveind] = maxwaveextra

                    self.binindices[n][s+1][i] = indarray

        self.speak('saving wavebin properties to the subcube')
        self.cube.subcube['wavebin'] = {}
        self.cube.subcube['wavebin']['binindices'] = self.binindices
        self.cube.subcube['wavebin']['wavelims'] = self.wavelims
        self.wavefiles = [str(i[0])+'-'+str(i[1]) for i in self.wavelims]
        self.cube.subcube['wavebin']['wavefiles'] = self.wavefiles
        np.save(self.inputs.saveas+'_subcube.npy', self.cube.subcube)


