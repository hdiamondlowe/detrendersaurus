import zachopy.Talker
import zachopy.Writer
Talker = zachopy.Talker.Talker
Writer = zachopy.Writer.Writer
import numpy as np
import os
from WaveBinner import WaveBinner
#import analysis.BatmanLC as BLC

class LCMaker(Talker, Writer):
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

        for w, wavefile in enumerate(self.wavebin.wavefiles):

            #file = self.inputs.nightname+'_'+str(wavelims[0])+'-'+str(wavelims[1])+'.npy'
            if self.inputs.nightname+'_'+wavefile+'.npy' in npyfiles: 
                self.speak(wavefile+' dictionary already exists in the detrender directory')
                pass

            else:
                #basename = os.path.splitext(wavefile)[0][14:]
                self.speak('creating dictionary for wavelength bin {0}'.format(wavefile))

                bininds = self.wavebin.binindices[:,:,w] # shape = (numexps, numstars, numwave)
                # create a dictionary for this wavelength bin
                bin = {}
                bin['wavelims'] = self.wavebin.wavelims[w]
                bin['bininds'] = bininds
                bin['binnedok'] = np.array([b for b in self.cube.subcube['ok']])

                # make a lightcurve to work off of
                raw_counts_targ = np.sum(self.cube.subcube['raw_counts'][self.inputs.target][self.inputs.targetpx] * bininds[:,0], 1) # shape = (numexps)
                raw_counts_targ = raw_counts_targ/np.mean(raw_counts_targ)
                raw_counts_comps = np.sum(np.sum([self.cube.subcube['raw_counts'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] * bininds[:, i+1] for i in range(len(self.inputs.comparison))], 0), 1)
                raw_counts_comps = raw_counts_comps/np.mean(raw_counts_comps)
                bin['lc'] = (raw_counts_targ/np.mean(raw_counts_targ))[self.cube.subcube['ok']]/(raw_counts_comps/np.mean(raw_counts_comps))[self.cube.subcube['ok']]

                # make a compcube that will be used for detrending
                bin['compcube'] = self.cube.makeCompCube(bininds)

                self.speak('creating output txt file for '+wavefile)

                # calculating photon noise limits
                raw_countsT = np.array(np.sum(self.cube.subcube['raw_counts'][self.inputs.target][self.inputs.targetpx] * bininds[:,0], 1)[self.cube.subcube['ok']])
                skyT = np.array(np.sum(self.cube.subcube['sky'][self.inputs.target][self.inputs.targetpx] * bininds[:,0], 1)[self.cube.subcube['ok']])
                raw_countsC = np.sum(np.array([np.sum(self.cube.subcube['raw_counts'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] * bininds[:,i+1], 1)[self.cube.subcube['ok']] for i in range(len(self.inputs.comparison))]), 0)
                skyC = np.sum(np.array([np.sum(self.cube.subcube['sky'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] * bininds[:,i+1], 1)[self.cube.subcube['ok']] for i in range(len(self.inputs.comparison))]), 0)
                sigmaT = np.mean(np.sqrt(raw_countsT+skyT)/raw_countsT)
                sigmaC = np.mean(np.sqrt(raw_countsC+skyC)/raw_countsC)
                sigmaF = np.sqrt(sigmaT**2 + sigmaC**2)

                # initiating writer for this particular wavelength bin output text file
                Writer.__init__(self, self.inputs.saveas+'_'+wavefile+'.txt')

                # write a bunch of stuff that you may want easy access too (without loading in the .npy file)
                self.write('output file for wavelength bin '+wavefile)
                if self.inputs.istarget == False: self.write('istarget = false. batman transit model will not be used!')
                self.write('starlist file: '+self.inputs.starlist)
                self.write('target: '+self.inputs.target)
                self.write('targetpx: '+self.inputs.targetpx)
                self.write('comparison: '+str(self.inputs.comparison))
                self.write('comparisonpx: '+str(self.inputs.comparisonpx))
                self.write('photon noise limits:')
                self.write('    target               comparison           T/C')
                self.write('    '+str(sigmaT)+'    '+str(sigmaC)+'    '+str(sigmaF))
                if self.inputs.optext: self.write('using optimally extracted spectra')
                self.write('fit labels: '+str(self.inputs.fitlabels))
                self.write('combining comparisons using:')
                if self.inputs.invvar: self.write('    inverse variance')
                else: self.write('    simple addition')
                self.write('tran labels: '+str(self.inputs.tranlabels))
                self.write('tran params: '+str(self.inputs.tranparams))
                self.write('tran bounds: '+str(self.inputs.tranbounds[0])+'\n             '+str(self.inputs.tranbounds[1]))

                # save the expected photon noise limit for the target/comparisons lightcurve
                bin['photnoiselim'] = sigmaF

                np.save(self.inputs.saveas+'_'+wavefile, bin)
                self.speak('saved dictionary for wavelength bin {0}'.format(wavefile))

