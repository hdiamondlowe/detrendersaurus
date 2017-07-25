import zachopy.Talker
import zachopy.Writer
Talker = zachopy.Talker.Talker
Writer = zachopy.Writer.Writer
import numpy as np
import os
import emcee
import sys
from emcee.utils import MPIPool
from ModelMaker import ModelMaker
from CubeReader import CubeReader
from Plotter import Plotter
#import emceehelper as mc

class MCFitter(Talker, Writer):

    '''this class will marginalize over the provided parameters using a levenberg-marquardt minimizer'''

    def __init__(self, detrender, wavefile):
        ''' initialize the lmfitter'''
        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.cube = self.detrender.cube
        self.wavefile = wavefile
        
        Writer.__init__(self, self.inputs.saveas+'_'+self.wavefile+'.txt')

        self.wavebin = np.load(self.inputs.saveas+'_'+self.wavefile+'.npy')[()]
        if 'mcfit' in self.wavebin.keys():
            self.speak('mcfit already exists for wavelength bin {0}'.format(self.wavefile))
        else: 
            self.speak('running mcfit for wavelength bin {0}'.format(self.wavefile))
            self.runMCFit()

    def runMCFit(self):

        self.paramvals = self.wavebin['lmfit']
        try: self.paramuncs = np.sqrt(np.diagonal(self.wavebin['linfit'].covar))
        except(ValueError):
            self.speak('the linear fit returned no uncertainties, consider changing tranbounds values')
            return

        self.write('mcmc uncertainty values:')
        [self.write('    '+self.inputs.paramlabels[i]+'_unc    '+str(self.paramuncs[i])) for i in range(len(self.inputs.paramlabels))] 
        if np.all(np.isfinite(np.array(self.paramuncs))) != True: 
            self.speak('mcmc error: there were non-finite uncertainties')
            return

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.inputs.parambounds[0]]
        self.mcmcbounds[1] = [i for i in self.inputs.parambounds[1]]
        
        for u in range(len(self.paramuncs)):
            if type(self.mcmcbounds[0][u]) == bool and self.mcmcbounds[0][u] == True: self.mcmcbounds[0][u] = self.paramvals[u]-self.paramuncs[u]*100.
            if type(self.mcmcbounds[1][u]) == bool and self.mcmcbounds[1][u] == True: self.mcmcbounds[1][u] = self.paramvals[u]+self.paramuncs[u]*100.
        self.write('lower and upper bounds for mcmc walkers:')
        self.write('    '+str(self.mcmcbounds[0])+'\n    '+str(self.mcmcbounds[1]))

        #this is a hack to try to make emcee picklable, not sure this helps at all...
        def makemodellocal(inputs, wavebin, p):
            modelobj = ModelMaker(inputs, wavebin, p)
            return modelobj.makemodel()

        def lnlike(p, lcb, inputs, wavebin):
            model = makemodellocal(inputs, wavebin, p)
            data_unc = (np.std(lcb - model))**2
            return -0.5*np.sum((lcb-model)**2/data_unc + np.log(2.*np.pi*data_unc))

        def lnprior(p):
            for i in range(len(p)):
                if not (self.mcmcbounds[0][i] <= p[i] <= self.mcmcbounds[1][i]):
                    return -np.inf
            return 0.0

        llvalues = []
        def lnprobfcn(p, lcb, inputs, wavebin):
            lp = lnprior(p)
            if not np.isfinite(lp):
                return -np.inf
            ll = lnlike(p, lcb, inputs, wavebin)
            llvalues.append(ll)
            return lp + ll


        ndim = len(self.paramvals)
        pos = [self.paramvals + self.paramuncs*1e-4*np.random.randn(ndim) for i in range(self.inputs.nwalkers)]
        self.speak('initiating emcee')
        self.sampler = emcee.EnsembleSampler(self.inputs.nwalkers, ndim, lnprobfcn, args=([self.wavebin['lc'], self.inputs, self.wavebin]))
        self.speak('running emcee')
        self.sampler.run_mcmc(pos, self.inputs.nsteps)
        #for i, result in enumerate(self.sampler.sample(pos, self.inputs.nsteps, np.ones((self.inputs.nwalkers, ndim)))):
        #    if (i+1) % 100 == 0:
        #        self.speak('{0:5.1%}'.format(float(i)/self.inputs.nsteps))


        #self.speak('sending parameters to emcee helper')
        #self.sampler, llvalues = mc.runemcee(self, self.inputs, self.cube, self.wavebin)

        self.mcchain = self.sampler.chain
        self.samples = self.sampler.chain[:,self.inputs.burnin:,:].reshape((-1, len(self.inputs.paramlabels)))
        self.mcparams = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(self.samples, [16, 50, 84], axis=0))))

        self.write('mcmc acceptance: '+str(np.median(self.sampler.acceptance_fraction)))

        self.write('mcmc params:')
        self.write('     parameter        value                  plus                  minus')
        [self.write('     '+self.inputs.paramlabels[i]+'     '+str(self.mcparams[i][0])+'     '+str(self.mcparams[i][1])+'     '+str(self.mcparams[i][2])) for i in range(len(self.inputs.paramlabels))]

        if 'dt' in self.inputs.paramlabels:
            ind = np.where(np.array(self.inputs.paramlabels) == 'dt')[0]
            self.inputs.t0 = (self.mcmc_params[ind][0] + self.inputs.toff)[0]
            self.speak('mcfit reseting t0 parameter, transit midpoint = {0}'.format(self.inputs.t0))
        self.write('mcfit transit midpoint: {0}'.format(self.inputs.t0))

        #calculate rms from mcfit
        modelobj = ModelMaker(self.inputs, self.wavebin, self.mcparams[:,0])
        model = modelobj.makemodel()
        resid = self.wavebin['lc'] - model
        data_unc = np.std(resid)          # standard devaition of the residuals from the first fit
        self.write('mcfit RMS: '+str(data_unc))

        # how many times the expected noise is the rms?
        self.write('x expected noise: {0}'.format(data_unc/self.wavebin['photnoiselim']))

        self.speak('saving mcfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['mcfit'] = self.mcparams
        self.wavebin['mcchain'] = self.mcchain
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)

        plot = Plotter(self.inputs, self.cube)
        plot.mcplots(self.wavebin)

        self.speak('done with mcfit for wavelength bin {0}'.format(self.wavefile))

