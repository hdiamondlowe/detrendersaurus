import zachopy.Talker
import zachopy.Writer
Talker = zachopy.Talker.Talker
Writer = zachopy.Writer.Writer
import numpy as np
import os
import emcee
import dynesty
from dynesty.dynamicsampler import stopping_function, weight_function
from dynesty.plotting import _quantile
import lmfit
import sys
from ldtk import LDPSetCreator, BoxcarFilter
from emcee.utils import MPIPool
from ModelMakerJoint import ModelMakerJoint
from CubeReaderJoint import CubeReaderJoint
from PlotterJoint import PlotterJoint
#import emceehelper as mc

class MCFitterJoint(Talker, Writer):

    '''this class will marginalize over the provided parameters using an mcmc'''

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
            if self.inputs.mcmccode == 'emcee':
                if self.wavebin['mcfit']['chain'].shape[1] == self.inputs.nsteps:
                    self.speak('mcfit has completed the number of steps')
                elif self.wavebin['mcfit']['chain'].shape[1] < self.inputs.nsteps:
                    self.speak('extending mcfit for more steps')
                    self.runMCFit_dynesty()
        else: 
            self.speak('running mcfit for wavelength bin {0}'.format(self.wavefile))
            if self.inputs.mcmccode == 'emcee': self.runMCFit_emcee()
            elif self.inputs.mcmccode == 'dynesty': self.runMCFit_dynesty()

    def runMCFit_emcee(self):

        if self.inputs.ldmodel:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.inputs.freeparambounds[0]]
        self.mcmcbounds[1] = [i for i in self.inputs.freeparambounds[1]]

        for u in range(len(self.inputs.freeparamnames)):
            if type(self.mcmcbounds[0][u]) == bool and self.mcmcbounds[0][u] == True: self.mcmcbounds[0][u] = self.wavebin['lmfit']['values'][u]-self.wavebin['lmfit']['uncs'][u]*100.
            if type(self.mcmcbounds[1][u]) == bool and self.mcmcbounds[1][u] == True: self.mcmcbounds[1][u] = self.wavebin['lmfit']['values'][u]+self.wavebin['lmfit']['uncs'][u]*100.
        self.write('lower and upper bounds for mcmc walkers:')
        for b, name in enumerate(self.inputs.freeparamnames):
            self.write('    '+name + '    '+str(self.mcmcbounds[0][b])+'    '+str(self.mcmcbounds[1][b]))

        #this is a hack to try to make emcee picklable, not sure this helps at all...
        def makemodellocal(inputs, wavebin, p):
            modelobj = ModelMakerJoint(inputs, wavebin, p)
            return modelobj.makemodel()

        def lnlike(p, lcb, inputs, wavebin):
            models = makemodellocal(inputs, wavebin, p)
            residuals = []
            for n, night in enumerate(self.inputs.nightname):
                residuals.append(self.wavebin['lc'][n] - models[n])
            residuals = np.hstack(residuals)
            data_unc = (np.std(residuals))**2
            return -0.5*np.sum((residuals)**2/data_unc + np.log(2.*np.pi*data_unc))

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

        ndim = len(self.inputs.freeparamnames)
        try: 
            if 'chain' in self.wavebin['mcfit'].keys():
                # want to run the remaining steps
                self.speak('starting mcfit from last step in already existing chain')
        except(KeyError):
            self.wavebin['mcfit'] = {}
            pos = [self.wavebin['lmfit']['values'] + self.wavebin['lmfit']['uncs']*1e-4*np.random.randn(ndim) for i in range(self.inputs.nwalkers)]
            self.speak('initiating emcee')
            self.sampler = emcee.EnsembleSampler(self.inputs.nwalkers, ndim, lnprobfcn, args=([self.wavebin['lc'], self.inputs, self.wavebin]))
            self.speak('running emcee')
            self.sampler.run_mcmc(pos, 100)
            self.speak('creating chain in mcfit dictionary')
            self.wavebin['mcfit']['chain'] = self.sampler.chain
            #np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)

        while self.wavebin['mcfit']['chain'].shape[1] < self.inputs.nsteps:
            pos = self.wavebin['mcfit']['chain'][:,-1,:]
            self.sampler = emcee.EnsembleSampler(self.inputs.nwalkers, ndim, lnprobfcn, args=([self.wavebin['lc'], self.inputs, self.wavebin]))
            for i, result in enumerate(self.sampler.sample(pos, iterations=1000)):
                if (i+1) == 1000:
                    self.wavebin['mcfit']['chain'] = np.append(self.wavebin['mcfit']['chain'], self.sampler.chain, axis=1)
                    self.speak('{0:5.1%}'.format(float(self.wavebin['mcfit']['chain'].shape[1])/self.inputs.nsteps))
                    self.speak('updating chain at step {0}'.format(self.wavebin['mcfit']['chain'].shape[1]))
                    try: np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)
                    except: 
                        print 'THERE WAS AN ERROR SAVING THE WAVEBIN'
                        return
                    try:
                        self.speak('autocorrelation array: {0}'.format(self.sampler.acor))
                    except: pass

        #self.speak('sending parameters to emcee helper')
        #self.sampler, llvalues = mc.runemcee(self, self.inputs, self.cube, self.wavebin)

        self.samples = self.wavebin['mcfit']['chain'][:,self.inputs.burnin:,:].reshape((-1, len(self.inputs.freeparamnames)))
        self.mcparams = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(self.samples, [16, 50, 84], axis=0))))

        # This is a goofy hack: I am using the lmfit Parameters class as storage for the mcmc values and uncertainties from emcee

        self.write('mcmc acceptance: '+str(np.median(self.sampler.acceptance_fraction)))

        self.write('mcmc params:')
        self.write('     parameter        value                  plus                  minus')
        [self.write('     '+self.inputs.freeparamnames[i]+'     '+str(self.mcparams[i][0])+'     '+str(self.mcparams[i][1])+'     '+str(self.mcparams[i][2])) for i in range(len(self.inputs.freeparamnames))]

        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                ind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                self.inputs.t0 = self.mcparams[ind] + self.inputs.toff
                self.speak('mcfit reseting t0 parameter for {0}, transit midpoint = {1}'.format(night, self.inputs.t0[n]))
            self.write('mcfit transit midpoint for {0}: {1}'.format(night, self.inputs.t0[n]))

        #calculate rms from mcfit
        modelobj = ModelMakerJoint(self.inputs, self.wavebin, self.mcparams[:,0])
        models = modelobj.makemodel()
        resid = []
        for n in range(len(self.inputs.nightname)):
            resid.append(self.wavebin['lc'][n] - models[n])
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('mcfit RMS: '+str(np.sqrt(np.sum(allresid**2)/len(allresid))))

        # how many times the expected noise is the rms?
        totphotnoise = []
        for n, night in enumerate(self.inputs.nightname):
            totphotnoise.append(self.wavebin['photnoiselim'][n]**2)
        totphotnoise = np.sqrt(np.sum(totphotnoise))
        self.write('total expected noise: {0}'.format(totphotnoise))
        self.write('x total expected noise: {0}'.format(data_unc/totphotnoise))

        self.speak('saving mcfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['mcfit']['values'] = self.mcparams
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)

        plot = PlotterJoint(self.inputs, self.cube)
        plot.mcplots(self.wavebin)

        self.speak('done with mcfit for wavelength bin {0}'.format(self.wavefile))

    def runMCFit_dynesty(self):

        if self.inputs.ldmodel:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.inputs.freeparambounds[0]]
        self.mcmcbounds[1] = [i for i in self.inputs.freeparambounds[1]]

        for u in range(len(self.inputs.freeparamnames)):
            if type(self.mcmcbounds[0][u]) == bool and self.mcmcbounds[0][u] == True: self.mcmcbounds[0][u] = self.wavebin['lmfit']['values'][u]-self.wavebin['lmfit']['uncs'][u]*10.
            if type(self.mcmcbounds[1][u]) == bool and self.mcmcbounds[1][u] == True: self.mcmcbounds[1][u] = self.wavebin['lmfit']['values'][u]+self.wavebin['lmfit']['uncs'][u]*10.
        self.write('lower and upper bounds for mcmc walkers:')
        for b, name in enumerate(self.inputs.freeparamnames):
            self.write('    '+name + '    '+str(self.mcmcbounds[0][b])+'    '+str(self.mcmcbounds[1][b]))

        def lnlike(p):
            modelobj = ModelMakerJoint(self.inputs, self.wavebin, p)
            models = modelobj.makemodel()
            residuals = []
            for n, night in enumerate(self.inputs.nightname):
                residuals.append(self.wavebin['lc'][n] - models[n])
            residuals = np.hstack(residuals)
            sigmasq= (np.std(residuals))**2
            logl = -0.5*np.sum((residuals)**2/sigmasq + np.log(2.*np.pi*sigmasq)) 
            return logl

        def ptform(p):
            x = np.array(p)
            for i in range(len(x)):
                span = self.mcmcbounds[1][i] - self.mcmcbounds[0][i]
                x[i] = x[i]*span + self.mcmcbounds[0][i]
            return x

        ndim = len(self.inputs.freeparamnames)

        self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, bound='multi', sample='slice', update_interval=float(ndim))
        self.dsampler.run_nested(nlive_init=int(5*ndim), nlive_batch=int(5*ndim), wt_kwargs={'pfrac': 1.0}) # place 100% of the weight on the posterior, don't sample the evidence

        quantiles = [_quantile(self.dsampler.results['samples'][:,i], [.16, .5, .84], weights=np.exp(self.dsampler.results['logwt'] - self.dsampler.results['logwt'][-1])) for i in range(len(self.inputs.freeparamnames))]
        self.mcparams = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), quantiles))

        self.speak('saving mcfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['mcfit'] = {}
        self.wavebin['mcfit']['results'] = self.dsampler.results
        self.wavebin['mcfit']['values'] = self.mcparams
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)

        self.write('mcmc params:')
        self.write('     parameter        value                  plus                  minus')
        [self.write('     '+self.inputs.freeparamnames[i]+'     '+str(self.mcparams[i][0])+'     '+str(self.mcparams[i][1])+'     '+str(self.mcparams[i][2])) for i in range(len(self.inputs.freeparamnames))]

        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                ind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                self.inputs.t0 = self.mcparams[ind] + self.inputs.toff
                self.speak('mcfit reseting t0 parameter for {0}, transit midpoint = {1}'.format(night, self.inputs.t0[n]))
            self.write('mcfit transit midpoint for {0}: {1}'.format(night, self.inputs.t0[n]))

        #calculate rms from mcfit
        modelobj = ModelMakerJoint(self.inputs, self.wavebin, self.mcparams[:,0])
        models = modelobj.makemodel()
        resid = []
        for n in range(len(self.inputs.nightname)):
            resid.append(self.wavebin['lc'][n] - models[n])
        allresid = np.hstack(resid)
        data_unc = np.std(allresid)
        self.write('mcfit RMS: '+str(np.sqrt(np.sum(allresid**2)/len(allresid))))

        # how many times the expected noise is the rms?
        totphotnoise = []
        for n, night in enumerate(self.inputs.nightname):
            totphotnoise.append(self.wavebin['photnoiselim'][n]**2)
        totphotnoise = np.sqrt(np.sum(totphotnoise))
        self.write('total expected noise: {0}'.format(totphotnoise))
        self.write('x total expected noise: {0}'.format(data_unc/totphotnoise))

        plot = PlotterJoint(self.inputs, self.cube)
        plot.mcplots(self.wavebin)

        self.speak('done with mcfit for wavelength bin {0}'.format(self.wavefile))

    def limbdarkparams(self, wavestart, waveend, teff=3270, teff_unc=104., 
                            logg=5.06, logg_unc=0.20, z=-0.12, z_unc=0.15):
        self.speak('using ldtk to derive limb darkening parameters')
        filters = BoxcarFilter('a', wavestart, waveend),     # Define passbands - Boxcar filters for transmission spectroscopy
        sc = LDPSetCreator(teff=(teff, teff_unc),             # Define your star, and the code
                           logg=(logg, logg_unc),             # downloads the uncached stellar 
                              z=(z   , z_unc),                # spectra from the Husser et al.
                        filters=filters)                      # FTP server automatically.

        ps = sc.create_profiles()                             # Create the limb darkening profiles
        u , u_unc = ps.coeffs_qd(do_mc=True)                  # Estimate non-linear law coefficients
        self.u0, self.u1 = u[0][0], u[0][1]
        self.write('limb darkening params: '+str(self.u0)+'  '+str(self.u1))

        for n in range(len(self.inputs.nightname)):
            if 'u0' in self.inputs.tranlabels[n]:
                self.inputs.tranparams[n][-2], self.inputs.tranparams[n][-1] = self.u0, self.u1
            else:
                # !Error! you also have to add to tranparambounds!
                self.inputs.tranlabels[n].append('u0')
                self.inputs.tranparams[n].append(self.u0)
                self.inputs.tranlabels[n].append('u1')
                self.inputs.tranparams[n].append(self.u1)
