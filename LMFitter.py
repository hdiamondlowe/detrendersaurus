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

        self.speak('running first lmfit')#, making output txt file')

        if self.inputs.ldmodel:
            self.limbdarkparams(self.wavebin['wavelims'][0]/10., self.wavebin['wavelims'][1]/10.)

        lmfitparams = lmfit.Parameters()
        for p in range(len(self.inputs.paramlabels)):
            lmfitparams[self.inputs.paramlabels[p]] = lmfit.Parameter(value=self.inputs.paramvalues[p])
        for u in range(len(self.inputs.paramlabels)):
            if type(self.inputs.parambounds[0][u]) == bool: minbound = None
            else: minbound = self.inputs.parambounds[0][u]
            if type(self.inputs.parambounds[1][u]) == bool: maxbound = None
            else: maxbound = self.inputs.parambounds[1][u]
            lmfitparams[self.inputs.paramlabels[u]].set(min=minbound, max=maxbound)

        def lineareqn(params):
            paramvals = [params[i].value for i in self.inputs.paramlabels]
            model = ModelMaker(self.inputs, self.wavebin, paramvals)
            return model.makemodel()

        def firstresiduals(params): 
            return (self.wavebin['lc'] - lineareqn(params))

        linfit1 = lmfit.minimize(firstresiduals, lmfitparams)
        self.lmparams1 = [linfit1.params[i].value for i in self.inputs.paramlabels]
        self.write('first lm params:')
        [self.write('    '+self.inputs.paramlabels[i]+'    '+str(self.lmparams1[i])) for i in range(len(self.inputs.paramlabels))]


        ######### do a second fit with priors, now that you know what the scatter is ########
        
        self.speak('running second lmfit after clipping >{0} sigma points'.format(self.inputs.sigclip))
        lmfitparams = lmfit.Parameters()
        for p in range(len(self.inputs.paramlabels)):
            lmfitparams[self.inputs.paramlabels[p]] = lmfit.Parameter(value=self.inputs.paramvalues[p])
        for u in range(len(self.inputs.paramlabels)):
            if type(self.inputs.parambounds[0][u]) == bool: minbound = None
            else: minbound = self.inputs.parambounds[0][u]
            if type(self.inputs.parambounds[1][u]) == bool: maxbound = None
            else: maxbound = self.inputs.parambounds[1][u]
            lmfitparams[self.inputs.paramlabels[u]].set(min=minbound, max=maxbound)
        
        # median absolute deviation sigma clipping to specified sigma value from inputs
        model = ModelMaker(self.inputs, self.wavebin, self.lmparams1)
        model = model.makemodel()
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
        self.wavebin['compcube'] = self.cube.makeCompCube(self.wavebin['bininds'], self.wavebin['binnedok'])
        self.write('clipped points: '+str(clip_start + clip_inds))
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)
        self.speak('remade lc and compcube for {0}'.format(self.wavefile))

        def dataresiduals(params):
            return (self.wavebin['lc'] - lineareqn(params))/data_unc
        def finalresiduals(params):
            return dataresiduals(params)
            #return np.hstack([dataresiduals(params), priort0(params)])#, prioru0(params), prioru1(params)])

        self.linfit2 = lmfit.minimize(finalresiduals, lmfitparams)
        self.lmparams2 = [self.linfit2.params[i].value for i in self.inputs.paramlabels]
        self.write('second round lm params:')
        [self.write('    '+self.inputs.paramlabels[i]+'    '+str(self.lmparams2[i])) for i in range(len(self.inputs.paramlabels))]

        if 'dt' in self.inputs.paramlabels:
            ind = int(np.where(np.array(self.inputs.paramlabels) == 'dt')[0])
            self.inputs.t0 = (self.lmparams2[ind] + self.inputs.toff)[0]
            self.speak('lmfit reseting t0 parameter, transit midpoint = {0}'.format(self.inputs.t0))
        self.write('lmfit transit midpoint: {0}'.format(self.inputs.t0))

        modelobj = ModelMaker(self.inputs, self.wavebin, self.lmparams2)
        model = modelobj.makemodel()
        resid = self.wavebin['lc'] - model
        data_unc = np.std(resid)
        self.write('lmfit SDNR: '+str(data_unc))  # this is the same as the rms!
        self.write('lmfit RMS: '+str(np.sqrt(np.sum(resid**2)/len(resid))))

        # how many times the expected noise is the rms?
        self.write('x expected noise: {0}'.format(data_unc/self.wavebin['photnoiselim']))

        # make BIC calculation
        # var = np.power(data_unc, 2.)
        var = 4.5e-7    # variance must remain fixed across all trials in order to make a comparison of BIC values
        lnlike = -0.5*np.sum((self.wavebin['lc']-model)**2/var + np.log(2.*np.pi*var))
        plbls = len(self.inputs.paramlabels)
        lnn = np.log(len(self.wavebin['lc']))
        BIC = -2.*lnlike + plbls*lnn
        self.write('BIC: '+str(BIC))

        self.speak('saving lmfit to wavelength bin {0}'.format(self.wavefile))
        self.wavebin['lmfit'] = self.lmparams2
        self.wavebin['linfit'] = self.linfit2
        np.save(self.inputs.saveas+'_'+self.wavefile, self.wavebin)

        plot = Plotter(self.detrender)
        plot.shiftstretchplot()
        plot.lmplots(self.wavebin)

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

