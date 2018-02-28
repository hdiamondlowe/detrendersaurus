import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np
import collections
from Plotter import Plotter

class CubeReaderJoint(Talker):
    ''' Reads in all the datacubes from the local subdirectories'''
    def __init__(self, detrender, jointdirectories):

        Talker.__init__(self)
        
        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.jointdirectories = jointdirectories

        try: 
            self.speak('trying to read in subcube')
            self.subcube = np.load(self.inputs.saveas+'_subcube.npy')[()]
            self.speak('loaded in subcube') 
        except(IOError):
            self.speak('subcube does not exist, creating a new one')
            self.subcube = []
            for n, subdir in enumerate(self.jointdirectories):
                self.n = n
                self.subdir = subdir
                self.datacubepath = self.subdir+'/trimmed_cube.npy'
                self.makeSubCube()
            np.save(self.inputs.saveas+'_subcube.npy', self.subcube)
            self.speak('subcube saved')

    def makeSubCube(self):

        self.speak('reading in datacube from {0} and extracting the arrays you care about'.format(self.subdir))

        cube = np.load(self.datacubepath)[()]
        ok = cube['temporal']['ok']

        bjd         = cube['temporal']['bjd']
        #self.wavelengths = cube['spectral']['wavelength']

        airmass     = cube['temporal']['airmass']               # (time)
        rotangle    = cube['temporal']['rotatore']              # (time)
        pwv         = cube['temporal']['pwv']                   # (time)
        norm        = np.ones(len(bjd))                         # for normalization constant

        centroid    = cube['squares']['centroid']               # [star][pix](time)
        width       = cube['squares']['width']                  # [star][pix](time)
        stretch     = cube['squares']['stretch']
        shift       = cube['squares']['shift']

        # if   optext == True:  
        #     print 'using optimal extraction raw counts'
        #    self.raw_counts  = cube['cubes']['raw_counts_optext']               # [star][pix](time, wave)
        raw_counts  = cube['cubes']['raw_counts']                      # [star][pix](time, wave)
        sky         = cube['cubes']['sky']                      # [star][pix](time, wave)
        dcentroid   = cube['cubes']['centroid']                 # [star][pix](time, wave)
        dwidth      = cube['cubes']['width']                    # [star][pix](time, wave)
        peak        = cube['cubes']['peak']                     # [star][pix](time, wave)
        wavelengths = cube['cubes']['wavelength_adjusted']
        #self.wavelengths_orig = cube['cubes']['wavelength_orig']

        # remaking the pwv parameters such that it is the residuals of a linear fit of pwv(t) = a + b*airmass(t); this is change in pwv minus the effects of airmass
        #z = np.polyfit(airmass, pwvraw, deg=1)
        #pfit = np.poly1d(z)
        #pwv = pwvraw-pfit(airmass)
        


        self.speak('making subcube of just the arrays you care about from the full cube from {0}'.format(self.subdir))
        
        subcube = {}
        subcube['ok'] = ok
        subcube['bjd'] = bjd
        subcube['airmass'] = airmass
        subcube['rotangle'] = rotangle
        subcube['pwv'] = pwv
        subcube['norm'] = norm
        
        # have to re-make these dictionaries
        subcube['centroid'] = collections.defaultdict(dict)
        subcube['width'] = collections.defaultdict(dict)
        subcube['stretch'] = collections.defaultdict(dict)
        subcube['shift'] = collections.defaultdict(dict)
        subcube['raw_counts'] = collections.defaultdict(dict)
        subcube['sky'] = collections.defaultdict(dict)
        subcube['dcentroid'] = collections.defaultdict(dict)
        subcube['dwidth'] = collections.defaultdict(dict)
        subcube['peak'] = collections.defaultdict(dict)
        subcube['wavelengths'] = collections.defaultdict(dict)

        subcube['centroid'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = centroid[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['width'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = width[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['stretch'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = stretch[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['shift'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = shift[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['raw_counts'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = raw_counts[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['sky'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = sky[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['dcentroid'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = dcentroid[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['dwidth'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = dwidth[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['peak'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = peak[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        subcube['wavelengths'][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] = wavelengths[self.inputs.target[self.n]][self.inputs.targetpx[self.n]]
        for i in range(len(self.inputs.comparison[self.n])):
            subcube['centroid'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = centroid[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['width'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = width[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['stretch'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = stretch[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['shift'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = shift[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['raw_counts'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = raw_counts[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['sky'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = sky[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['dcentroid'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = dcentroid[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['dwidth'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = dwidth[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['peak'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = peak[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]
            subcube['wavelengths'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] = wavelengths[self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]]

        self.subcube.append(subcube)

    def makeCompCube(self, subbinindices, n, *binnedok):
        '''A minicube is a subset of a subcube that only includes the relevant wavelength information for a given wavelength bin'''

        self.speak('making compcube for joint subdirectory number {0}'.format(n))
        #subbinindices are of the format [numexps, numwave]
        self.subbinindices = subbinindices
        self.n = n

        if binnedok: self.binnedok = binnedok[0]
        else: self.binnedok = np.array([b for b in self.subcube[self.n]['ok']])


        self.compcube = {}
        self.compcube['binnedok'] = self.binnedok
        self.compcube['bjd'] = self.subcube[self.n]['bjd'][self.binnedok]
        self.compcube['norm'] = self.subcube[self.n]['norm'][self.binnedok]

        for key in ['airmass', 'rotangle', 'pwv']:
            self.compcube[key] = (self.subcube[self.n][key][self.binnedok] - np.mean(self.subcube[self.n][key][self.binnedok]))/(np.std(self.subcube[self.n][key][self.binnedok]))

        if self.inputs.invvar: 
            self.speak('weighting by inverse variance')
            raw_counts_comps = np.array([np.sum(self.subcube[self.n]['raw_counts'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] * self.subindices, 1)[self.binnedok] for i in range(len(self.inputs.comparison[self.n]))])
            sky_counts_comps = np.array([np.sum(self.subcube[self.n]['sky'][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] * self.subindices, 1)[self.binnedok] for i in range(len(self.inputs.comparison[self.n]))])
            sig2 = raw_counts_comps + sky_counts_comps
            den = np.sum((1./sig2), 0)

        for key in ['centroid', 'width', 'stretch', 'shift']:
            #keyarray = np.array([self.subcube[self.n][key][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]][self.binnedok] for i in range(len(self.inputs.comparison[self.n]))])
            if self.inputs.invvar:
                num = np.sum((keyarray/sig2), 0)
                self.compcube[key] = num/den
            else:
                #summed = np.sum(keyarray, 0)
                #self.compcube[key] = (summed - np.mean(summed))/np.std(summed)
                # test if parameter should be difference between GJ 1132 and key
                #summed = self.subcube[self.n][key][self.inputs.target[self.n]][self.inputs.targetpx[self.n]][self.binnedok] - np.sum(keyarray, 0)
                # instead use GJ 1132 parameters to detrend against
                keyarray = self.subcube[self.n][key][self.inputs.target[self.n]][self.inputs.targetpx[self.n]][self.binnedok]
                self.compcube[key] = (keyarray - np.mean(keyarray))/np.std(keyarray)

        for key in ['raw_counts', 'sky', 'dcentroid', 'dwidth', 'peak']:
            #keyarray = np.array([np.sum(self.subcube[self.n][key][self.inputs.comparison[self.n][i]][self.inputs.comparisonpx[self.n][i]] * self.subbinindices[:, i+1], 1)[self.binnedok] for i in range(len(self.inputs.comparison[self.n]))])
            if self.inputs.invvar:
                num = np.sum((keyarray/sig2), 0)
                self.compcube[key] = num/den
            else: 
                #summed = np.sum(keyarray, 0)
                # test if parameter should be difference between GJ 1132 and key
                #summed = np.sum(self.subcube[self.n][key][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] * self.subbinindices[:, i+1], 1)[self.binnedok] - np.sum(keyarray, 0)
                #self.compcube[key] = (summed - np.mean(summed))/np.std(summed)
                # instead use GJ 1132 parameters to detrend against
                keyarray = np.sum(self.subcube[self.n][key][self.inputs.target[self.n]][self.inputs.targetpx[self.n]] * self.subbinindices[:, 0], 1)[self.binnedok]
                self.compcube[key] = (keyarray - np.mean(keyarray))/np.std(keyarray)

        return self.compcube
