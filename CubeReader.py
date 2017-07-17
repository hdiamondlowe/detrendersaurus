import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np
import collections

class CubeReader(Talker):
    ''' Reads in the datacube in the local directory'''
    def __init__(self, detrender):

        Talker.__init__(self)
        
        self.detrender = detrender
        self.inputs = self.detrender.inputs

        try: 
            self.subcube = np.load(self.inputs.saveas+'_subcube.npy')[()]
            self.speak('loaded in subcube')
        except(IOError):
            self.readCube()
            self.makeSubCube()

    def readCube(self):

        self.speak('reading in datacube and extracting the arrays you care about')

        cube = np.load(self.detrender.datacubepath)[()]
        self.ok = cube['temporal']['ok']

        self.bjd         = cube['temporal']['bjd']
        #self.wavelengths = cube['spectral']['wavelength']

        self.airmass     = cube['temporal']['airmass']               # (time)
        self.rotangle    = cube['temporal']['rotatore']              # (time)
        self.pwv         = cube['temporal']['pwv']                   # (time)
        self.norm        = np.ones(len(self.bjd))                         # for normalization constant

        self.centroid    = cube['squares']['centroid']               # [star][pix](time)
        self.width       = cube['squares']['width']                  # [star][pix](time)
        self.stretch     = cube['squares']['stretch']
        self.shift       = cube['squares']['shift']

        # if   optext == True:  
        #     print 'using optimal extraction raw counts'
        #    self.raw_counts  = cube['cubes']['raw_counts_optext']               # [star][pix](time, wave)
        self.raw_counts  = cube['cubes']['raw_counts']                      # [star][pix](time, wave)
        self.sky         = cube['cubes']['sky']                      # [star][pix](time, wave)
        self.dcentroid   = cube['cubes']['centroid']                 # [star][pix](time, wave)
        self.dwidth      = cube['cubes']['width']                    # [star][pix](time, wave)
        self.peak        = cube['cubes']['peak']                     # [star][pix](time, wave)
        self.wavelengths = cube['cubes']['wavelength_adjusted']
        #self.wavelengths_orig = cube['cubes']['wavelength_orig']

    def makeSubCube(self):

        self.speak('making subcube of just the arrays you care about from the full cube')
        
        self.subcube = {}
        self.subcube['ok'] = self.ok
        self.subcube['bjd'] = self.bjd
        self.subcube['airmass'] = self.airmass
        self.subcube['rotangle'] = self.rotangle
        self.subcube['pwv'] = self.pwv
        self.subcube['norm'] = self.norm
        
        # have to re-make these dictionaries
        self.subcube['centroid'] = collections.defaultdict(dict)
        self.subcube['width'] = collections.defaultdict(dict)
        self.subcube['stretch'] = collections.defaultdict(dict)
        self.subcube['shift'] = collections.defaultdict(dict)
        self.subcube['raw_counts'] = collections.defaultdict(dict)
        self.subcube['sky'] = collections.defaultdict(dict)
        self.subcube['dcentroid'] = collections.defaultdict(dict)
        self.subcube['dwidth'] = collections.defaultdict(dict)
        self.subcube['peak'] = collections.defaultdict(dict)
        self.subcube['wavelengths'] = collections.defaultdict(dict)

        self.subcube['centroid'][self.inputs.target][self.inputs.targetpx] = self.centroid[self.inputs.target][self.inputs.targetpx]
        self.subcube['width'][self.inputs.target][self.inputs.targetpx] = self.width[self.inputs.target][self.inputs.targetpx]
        self.subcube['stretch'][self.inputs.target][self.inputs.targetpx] = self.stretch[self.inputs.target][self.inputs.targetpx]
        self.subcube['shift'][self.inputs.target][self.inputs.targetpx] = self.shift[self.inputs.target][self.inputs.targetpx]
        self.subcube['raw_counts'][self.inputs.target][self.inputs.targetpx] = self.raw_counts[self.inputs.target][self.inputs.targetpx]
        self.subcube['sky'][self.inputs.target][self.inputs.targetpx] = self.sky[self.inputs.target][self.inputs.targetpx]
        self.subcube['dcentroid'][self.inputs.target][self.inputs.targetpx] = self.dcentroid[self.inputs.target][self.inputs.targetpx]
        self.subcube['dwidth'][self.inputs.target][self.inputs.targetpx] = self.peak[self.inputs.target][self.inputs.targetpx]
        self.subcube['peak'][self.inputs.target][self.inputs.targetpx] = self.peak[self.inputs.target][self.inputs.targetpx]
        self.subcube['wavelengths'][self.inputs.target][self.inputs.targetpx] = self.wavelengths[self.inputs.target][self.inputs.targetpx]
        for i in range(len(self.inputs.comparison)):
            self.subcube['centroid'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.centroid[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['width'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.width[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['stretch'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.stretch[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['shift'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.shift[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['raw_counts'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.raw_counts[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['sky'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.sky[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['dcentroid'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.dcentroid[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['dwidth'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.peak[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['peak'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.peak[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]
            self.subcube['wavelengths'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] = self.wavelengths[self.inputs.comparison[i]][self.inputs.comparisonpx[i]]

        np.save(self.inputs.saveas+'_subcube.npy', self.subcube)
        self.speak('subcube saved')

    def makeCompCube(self, subbinindices, *binnedok):
        '''A minicube is a subset of a subcube that only includes the relevant wavelength information for a given wavelength bin'''

        self.speak('making compcube')
        #subbinindices are of the format [numexps, numwave]
        self.subbinindices = subbinindices

        if binnedok: self.binnedok = binnedok[0]
        else: self.binnedok = np.array([b for b in self.subcube['ok']])


        self.compcube = {}
        self.compcube['binnedok'] = self.binnedok
        self.compcube['bjd'] = self.subcube['bjd'][self.binnedok]

        for key in ['airmass', 'rotangle', 'pwv', 'norm']:
            self.compcube[key] = (self.subcube[key][self.binnedok] - np.mean(self.subcube[key][self.binnedok]))/(np.std(self.subcube[key][self.binnedok]))

        if self.inputs.invvar: 
            self.speak('weighting by inverse variance')
            raw_counts_comps = np.array([np.sum(self.subcube['raw_counts'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] * self.subindices, 1)[self.binnedok] for i in range(len(self.inputs.comparison))])
            sky_counts_comps = np.array([np.sum(self.subcube['sky'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] * self.subindices, 1)[self.binnedok] for i in range(len(self.inputs.comparison))])
            sig2 = self.raw_counts_comps + self.sky_counts_comps
            den = np.sum((1./sig2), 0)

        for key in ['centroid', 'width', 'stretch', 'shift']:
            keyarray = np.array([self.subcube[key][self.inputs.comparison[i]][self.inputs.comparisonpx[i]][self.binnedok] for i in range(len(self.inputs.comparison))])
            if self.inputs.invvar:
                num = np.sum((keyarray/sig2), 0)
                self.compcube[key] = num/den
            else: 
                summed = np.sum(keyarray, 0)
                self.compcube[key] = (summed - np.mean(summed))/np.std(summed)

        for key in ['raw_counts', 'sky', 'dcentroid', 'dwidth', 'peak']:
            keyarray = np.array([np.sum(self.subcube[key][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] * self.subbinindices[:, i+1], 1)[self.binnedok] for i in range(len(self.inputs.comparison))])
            if self.inputs.invvar:
                num = np.sum((keyarray/sig2), 0)
                self.compcube[key] = num/den
            else: 
                summed = np.sum(keyarray, 0)
                self.compcube[key] = (summed - np.mean(summed))/np.std(summed)

        return self.compcube
