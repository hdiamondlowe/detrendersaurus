import zachopy.Talker
import zachopy.Writer
Talker = zachopy.Talker.Talker
Writer = zachopy.Writer.Writer
import numpy as np
import lmfit
import os
from ldtk import LDPSetCreator, BoxcarFilter
from ModelMaker import ModelMaker
from CubeReader import CubeReader
from Plotter import Plotter

class LMFitter(Talker, Writer):

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
        if 'lmfit' in self.wavebin.keys():
            self.speak('lmfit already exists for wavelength bin {0}'.format(self.wavefile))
        else: 
            self.speak('running lmfit for wavelength bin {0}'.format(self.wavefile))
            self.runLMFit()

    def runLMFit(self):

        if self.inputs.ldmodel:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        self.speak('running first joint lmfit scaling by photon noise limits')#, making output txt file')

        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.inputs.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.inputs.freeparamvalues[n])
            if self.inputs.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.inputs.freeparambounds[0][n]
            if self.inputs.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.inputs.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        def lineareqn(params):
            paramvals = [params[name].value for name in self.inputs.freeparamnames]
            model = ModelMaker(self.inputs, self.wavebin, paramvals)
            return model.makemodel()

        def residuals1(params):
            model = lineareqn(params)
            residuals = (self.wavebin['lc'] - model)/self.wavebin['photnoiseest'] # weight by photon noise limit (expected noise)
            return residuals

        fit_kws={'epsfcn':1e-5}  # set the stepsize to something small but reasonable; withough this lmfit may have trouble perturbing values
            #, 'full_output':True, 'xtol':1e-5, 'ftol':1e-5, 'gtol':1e-5}
        self.linfit1 = lmfit.minimize(fcn=residuals1, params=lmfitparams, method='leastsq', **fit_kws)
        self.write('1st lm params:')
        [self.write('    '+name+'    '+str(self.linfit1.params[name].value)) for name in self.inputs.freeparamnames]


        ######### do a second fit with priors, now that you know what the initial scatter is ########
        
        self.speak('running second lmfit after clipping >{0} sigma points'.format(self.inputs.sigclip))
 
        # median absolute deviation sigma clipping to specified sigma value from inputs
        linfit1paramvals = [self.linfit1.params[name].value for name in self.inputs.freeparamnames]
        modelobj = ModelMaker(self.inputs, self.wavebin, linfit1paramvals)
        model = modelobj.makemodel()
        resid = self.wavebin['lc'] - model
        mad = np.median(abs(resid - np.median(resid)))
        scale = 1.4826
        data_unc = scale*mad               # scale x median absolute deviation
        clip_inds = np.where((resid > (self.inputs.sigclip*data_unc)) | (resid < (-self.inputs.sigclip*data_unc)))[0]
        clip_start = np.where(self.wavebin['binnedok'])[0][0]
        self.wavebin['binnedok'][clip_start + clip_inds] = False
        # need to update wavebin lc and compcube to reflect data clipping
        newbinnedok = np.ones(self.wavebin['lc'].shape, dtype=bool)
        newbinnedok[clip_inds] = False
        self.wavebin['lc'] = self.wavebin['lc'][newbinnedok]
        self.wavebin['photnoiseest'] = self.wavebin['photnoiseest'][newbinnedok]
        self.wavebin['compcube'] = self.cube.makeCompCube(self.wavebin['bininds'], self.wavebin['binnedok'])
        self.write('clipped points: {0}'.format(clip_start+clip_inds))
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)
        self.speak('remade lc and compcube for wavebin {0}'.format(self.wavefile))

        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.inputs.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.inputs.freeparamvalues[n])
            if self.inputs.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.inputs.freeparambounds[0][n]
            if self.inputs.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.inputs.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        def residuals2(params):
            model = lineareqn(params)
            residuals = (self.wavebin['lc'] - model)/self.wavebin['photnoiseest'] # weight by photon noise limit (expected noise)
            return residuals

        self.linfit2 = lmfit.minimize(fcn=residuals2, params=lmfitparams, method='leastsq', **fit_kws)
        self.write('2nd lm params:')
        [self.write('    '+name+'    '+str(self.linfit2.params[name].value)) for name in self.inputs.freeparamnames]

        ######### do a third fit, now with calculated uncertainties ########
        
        self.speak('running third lmfit after calculating undertainties from the data'.format(self.inputs.sigclip))
 
        lmfitparams = lmfit.Parameters()
        for n, name in enumerate(self.inputs.freeparamnames):
            lmfitparams[name] = lmfit.Parameter(value=self.inputs.freeparamvalues[n])
            if self.inputs.freeparambounds[0][n] == True: minbound = None
            else: minbound = self.inputs.freeparambounds[0][n]
            if self.inputs.freeparambounds[1][n] == True: maxbound = None
            else: maxbound = self.inputs.freeparambounds[1][n]
            lmfitparams[name].set(min=minbound, max=maxbound)

        linfit2paramvals = [self.linfit2.params[name].value for name in self.inputs.freeparamnames]
        modelobj = ModelMaker(self.inputs, self.wavebin, linfit2paramvals)
        model = modelobj.makemodel()
        resid = self.wavebin['lc'] - model
        data_unc = np.std(resid)

        def residuals3(params):
            model = lineareqn(params)
            residuals = (self.wavebin['lc'] - model)/data_unc # weight by calculated uncertainty
            return residuals

        self.linfit3 = lmfit.minimize(fcn=residuals3, params=lmfitparams, method='leastsq', **fit_kws)
        self.write('3rd lm params:')
        [self.write('    '+name+'    '+str(self.linfit3.params[name].value)) for name in self.inputs.freeparamnames]
        #[[self.write('    '+flabel+str(n)+'    '+str(self.linfit3.params[flabel+str(n)].value)) for flabel in self.inputs.fitlabels[n]] for n in range(len(self.inputs.nightname))]
        #[self.write('    '+tlabel+'    '+str(self.linfit3.params[tlabel].value)) for tlabel in self.freetranlabels]

        if 'dt' in self.linfit3.params.keys():
            self.inputs.t0 = self.linfit3.params['dt'] + self.inputs.toff
            self.speak('lmfit reseting t0 parameter, midpoint = {0}'.format(self.inputs.t0))
        self.write('lmfit transit midpoint: {0}'.format(self.inputs.t0))

        linfit3paramvals = [self.linfit3.params[name].value for name in self.inputs.freeparamnames]
        modelobj = ModelMaker(self.inputs, self.wavebin, linfit3paramvals)
        model = modelobj.makemodel()
        resid = self.wavebin['lc'] - model
        data_unc = np.std(resid)
        self.write('lmfit SDNR: '+str(data_unc))  # this is the same as the rms!

        # how many times the expected noise is the rms?
        self.write('x expected noise: {0}'.format(data_unc/np.mean(self.wavebin['photnoiseest'])))

        # make BIC calculations
        # var = np.power(data_unc, 2.)
        var = 4.5e-7    # variance must remain fixed across all trials in order to make a comparison of BIC values
        lnlike = -0.5*np.sum((self.wavebin['lc'] - model)**2/var + np.log(2.*np.pi*var))
        plbls = len(self.inputs.freeparamnames)
        lnn = np.log(len(self.wavebin['lc']))
        BIC = -2.*lnlike + plbls*lnn
        self.write('BIC: {0}'.format(BIC))

        self.speak('saving lmfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['lmfit'] = {}
        self.wavebin['lmfit']['values'] = linfit3paramvals
        try: self.wavebin['lmfit']['uncs'] = np.sqrt(np.diagonal(self.linfit3.covar))
        except(ValueError):
            self.speak('the linear fit returned no uncertainties, consider changing tranbounds values')
            return
        if not np.all(np.isfinite(np.array(self.wavebin['lmfit']['uncs']))): 
            self.speak('lmfit error: there were non-finite uncertainties')
            return
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)

        plot = Plotter(self.inputs, self.cube)
        plot.lmplots(self.wavebin, [self.linfit1, self.linfit2, self.linfit3])

        self.speak('done with lmfit for wavelength bin {0}'.format(self.wavefile))

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

