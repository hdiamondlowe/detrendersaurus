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
import sys
from ldtk import LDPSetCreator, BoxcarFilter
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
            if self.inputs.mcmccode == 'emcee': self.runMCFit_emcee()
            elif self.inputs.mcmccode == 'dynesty': self.runMCFit_dynesty()

    def runMCFit_emcee(self):

        if self.inputs.ldmodel:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.inputs.freeparambounds[0]]
        self.mcmcbounds[1] = [i for i in self.inputs.freeparambounds[1]]

        for u in range(len(self.inputs.freeparamnames)):
            if type(self.mcmcbounds[0][u]) == bool and self.mcmcbounds[0][u] == True: self.mcmcbounds[0][u] = self.wavebin['lmfit']['values'][u]-self.wavebin['lmfit']['uncs'][u]*25.
            if type(self.mcmcbounds[1][u]) == bool and self.mcmcbounds[1][u] == True: self.mcmcbounds[1][u] = self.wavebin['lmfit']['values'][u]+self.wavebin['lmfit']['uncs'][u]*25.
        self.write('lower and upper bounds for mcmc walkers:')
        for b, name in enumerate(self.inputs.freeparamnames):
            self.write('    '+name + '    '+str(self.mcmcbounds[0][b])+'    '+str(self.mcmcbounds[1][b]))

        def lnlike(p):
            modelobj = ModelMaker(self.inputs, self.wavebin, p)
            model = modelobj.makemodel()
            data_unc = (np.std(self.wavebin['lc'] - model))**2
            return -0.5*np.sum((self.wavebin['lc']-model)**2/data_unc + np.log(2.*np.pi*data_unc))

        def lnprior(p):
            for i in range(len(p)):
                if not (self.mcmcbounds[0][i] <= p[i] <= self.mcmcbounds[1][i]):
                    return -np.inf
            return 0.0

        self.llvalues = []
        def lnprobfcn(p):
            lp = lnprior(p)
            if not np.isfinite(lp):
                return -np.inf
            ll = lnlike(p)
            self.llvalues.append(ll)
            return lp + ll


        ndim = len(self.inputs.freeparamnames)
        pos = [self.wavebin['lmfit']['values'] + self.wavebin['lmfit']['uncs']*1e-4*np.random.randn(ndim) for i in range(self.inputs.nwalkers)]
        self.speak('initiating emcee')
        self.sampler = emcee.EnsembleSampler(self.inputs.nwalkers, ndim, lnprobfcn)
        self.speak('running emcee')
        self.sampler.run_mcmc(pos, self.inputs.nsteps)
        #for i, result in enumerate(self.sampler.sample(pos, self.inputs.nsteps, np.ones((self.inputs.nwalkers, ndim)))):
        #    if (i+1) % 100 == 0:
        #        self.speak('{0:5.1%}'.format(float(i)/self.inputs.nsteps))


        #self.speak('sending parameters to emcee helper')
        #self.sampler, llvalues = mc.runemcee(self, self.inputs, self.cube, self.wavebin)

        self.mcchain = self.sampler.chain
        self.samples = self.sampler.chain[:,self.inputs.burnin:,:].reshape((-1, len(self.inputs.freeparamnames)))
        self.mcparams = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(self.samples, [16, 50, 84], axis=0))))

        self.write('mcmc acceptance: '+str(np.median(self.sampler.acceptance_fraction)))

        self.write('mcmc params:')
        self.write('     parameter        value                  plus                  minus')
        [self.write('     '+self.inputs.freeparamnames[i]+'     '+str(self.mcparams[i][0])+'     '+str(self.mcparams[i][1])+'     '+str(self.mcparams[i][2])) for i in range(len(self.inputs.freeparamnames))]

        if 'dt' in self.inputs.freeparamnames:
            ind = np.where(np.array(self.inputs.freeparamnames) == 'dt')[0]
            self.inputs.t0 = (self.mcparams[ind][0] + self.inputs.toff)[0]
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
        self.wavebin['mcfit'] = {}
        self.wavebin['mcfit']['values'] = self.mcparams
        self.wavebin['mcfit']['chain'] = self.mcchain
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)

        plot = Plotter(self.inputs, self.cube)
        plot.mcplots(self.wavebin)

        self.speak('done with mcfit for wavelength bin {0}'.format(self.wavefile))

    def runMCFit_dynesty(self):

        if self.inputs.ldmodel:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        # add the 's' scaling parameter into the fit
        if 's' in self.inputs.freeparamnames: pass
        else:
            self.inputs.freeparamnames.append('s')
            self.inputs.freeparamvalues.append(1)
            self.inputs.freeparambounds[0].append(0.01)
            self.inputs.freeparambounds[1].append(10.)

        self.mcmcbounds = [[],[]]
        self.mcmcbounds[0] = [i for i in self.inputs.freeparambounds[0]]
        self.mcmcbounds[1] = [i for i in self.inputs.freeparambounds[1]]

        for u in range(len(self.inputs.freeparamnames)):
            if type(self.mcmcbounds[0][u]) == bool and self.mcmcbounds[0][u] == True: self.mcmcbounds[0][u] = self.wavebin['lmfit']['values'][u]-self.wavebin['lmfit']['uncs'][u]*10.
            if type(self.mcmcbounds[1][u]) == bool and self.mcmcbounds[1][u] == True: self.mcmcbounds[1][u] = self.wavebin['lmfit']['values'][u]+self.wavebin['lmfit']['uncs'][u]*10.
        self.write('lower and upper bounds for mcmc walkers:')
        for b, name in enumerate(self.inputs.freeparamnames):
            self.write('    '+name + '    '+str(self.mcmcbounds[0][b])+'    '+str(self.mcmcbounds[1][b]))

        # rescaling uncertainties as a free parameter during the fit (Berta, et al. 2011, references therein)
        def lnlike(p):
            modelobj = ModelMaker(self.inputs, self.wavebin, p)
            model = modelobj.makemodel()

            # p[sind] is an 's' parameter; if the uncertainties do not need to be re-scaled then s = 1
            # there is a single 's' parameter for each night's fit - helpful if a dataset is far from the photon noise
            sind = int(np.where(np.array(self.inputs.freeparamnames) == 's')[0])
            penaltyterm = -len(self.wavebin['photnoiseest']) * np.log(p[sind])
            chi2 = ((self.wavebin['lc'] - model)/self.wavebin['photnoiseest'])**2
            logl = penaltyterm - 0.5*(1./(p[sind]**2))*np.sum(chi2)
            return logl

        def ptform(p):
            x = np.array(p)
            for i in range(len(x)):
                span = self.mcmcbounds[1][i] - self.mcmcbounds[0][i]
                x[i] = x[i]*span + self.mcmcbounds[0][i]
            return x

        ndim = len(self.inputs.freeparamnames)

        self.dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim=ndim, bound='multi', sample='slice')
        self.dsampler.run_nested(wt_kwargs={'pfrac': 1.0})

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

        if 'dt' in self.inputs.freeparamnames:
            ind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt')[0])
            self.inputs.t0 = self.mcparams[ind] + self.inputs.toff
            self.speak('mcfit reseting t0, transit midpoint = {0}'.format(self.inputs.t0))
        self.write('mcfit transit midpoint: {0}'.format(self.inputs.t0))

        #calculate rms from mcfit
        modelobj = ModelMaker(self.inputs, self.wavebin, self.mcparams[:,0])
        model = modelobj.makemodel()
        resid = self.wavebin['lc'] - model
        data_unc = np.std(resid)
        self.write('mcfit RMS: '+str(data_unc))

        # how many times the expected noise is the rms?
        self.write('x total expected noise: {0}'.format(data_unc/np.mean(self.wavebin['photnoiseest'])))

        plot = Plotter(self.inputs, self.cube)
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

        if 'u0' in self.inputs.tranlabels:
            self.inputs.tranparams[-2], self.inputs.tranparams[-1] = self.u0, self.u1
        else:
            self.inputs.tranlabels.append('u0')
            self.inputs.tranparams.append(self.u0)
            self.inputs.tranlabels.append('u1')
            self.inputs.tranparams.append(self.u1)



