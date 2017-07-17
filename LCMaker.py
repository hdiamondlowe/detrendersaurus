import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np
import os
from WaveBinner import WaveBinner
#import analysis.BatmanLC as BLC

class LCMaker(Talker):
    '''LCMaker object trims the data in time and also creates light curves for each wavelength bin and saves them in their own .npy files.'''    
    def __init__(self, detrender):
        '''Initialize a LCMaker object.'''

        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.cube = self.detrender.cube

        self.trimTimeSeries()
        self.makeBinnedLCs()

    def trimTimeSeries(self):
        '''trim the baseline of the time series to 1.5 * transit duration on either side of the transit midpoint'''
        # trim the light curves such that there is 1 transit duration on either side of the transit, and not more
        # Tdur needs to be in days
        self.speak('trimming excess baseline')
        outside_transit = np.where((self.cube.subcube['bjd'] < self.inputs.t0-(1.5*self.inputs.Tdur)) | (self.cube.subcube['bjd'] > self.inputs.t0+(1.5*self.inputs.Tdur)))
        # !!! may have issue here when changing the midpoint time; may need to reset self.ok to all True and then add in time clip
        self.cube.subcube['ok'][outside_transit] = False

    def makeBinnedLCs(self):

        self.wavebin = WaveBinner(self.detrender)

        npyfiles = []
        for file in os.listdir(self.detrender.directoryname):
            if file.endswith('.npy'):
                npyfiles.append(file)

        for w, wavelims in enumerate(self.wavebin.wavelims):

            file = self.inputs.nightname+'_'+str(wavelims[0])+'-'+str(wavelims[1])+'.npy'
            if file in npyfiles: 
                self.speak(file+' already exists in the detrend directory')
                pass

            else:
                self.speak('creating dictionary for wavelength bin {0}-{1}'.format(wavelims[0], wavelims[1]))

                bininds = self.wavebin.binindices[:,:,w] # shape = (numexps, numstars, numwave)
                # create a dictionary for this wavelength bin
                bin = {}
                bin['binnedok'] = np.array([b for b in self.cube.subcube['ok']])

                raw_counts_targ = np.sum(self.cube.subcube['raw_counts'][self.inputs.target][self.inputs.targetpx] * bininds[:,0], 1) # shape = (numexps)
                raw_counts_targ = raw_counts_targ/np.mean(raw_counts_targ)
                raw_counts_comps = np.sum(np.sum([self.cube.subcube['raw_counts'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] * bininds[:, i+1] for i in range(len(self.inputs.comparison))], 0), 1)
                raw_counts_comps = raw_counts_comps/np.mean(raw_counts_comps)
                bin['lc'] = (raw_counts_targ/np.mean(raw_counts_targ))[self.cube.subcube['ok']]/(raw_counts_comps/np.mean(raw_counts_comps))[self.cube.subcube['ok']]

                # make a compcube that will be used for detrending
                bin['compcube'] = self.cube.makeCompCube(bininds)

                np.save(self.inputs.saveas+'_'+str(wavelims[0])+'-'+str(wavelims[1]), bin)
                self.speak('saved dictionary for wavelength bin {0}-{1}'.format(wavelims[0], wavelims[1]))

        #self.cube.makeMiniCube()

        #self.targcomp_binned = WBA.WavelengthBinnerAbridged(self.reducer.bjd, self.toff+self.tranparams[0], self.Tdur, self.raw_counts, self.sky, self.wavelengths, self.wavelengths_orig, self.targ, self.targpx, self.comp, self.comppx, self.wavelength_lims, masktarget=False, invvar=self.inputs.invvar) 
        #self.targcomp_binned.bin_wavelengths_adjusted(bin_len)
        #self.targcomp_binned.get_binned_lcs()

        #self.keys = np.sort(self.targcomp_binned.binned_lcs_dict.keys())
        #self.istarget = istarget
