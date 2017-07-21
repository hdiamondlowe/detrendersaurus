import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np
import os
from ModelMaker import ModelMaker
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner

class Plotter(Talker):

    '''this class will plot all the things you wish to see'''

    def __init__(self, detrender):
        ''' initialize the plotter'''
        Talker.__init__(self)

        self.detrender = detrender
        self.inputs = self.detrender.inputs
        self.cube = self.detrender.cube

    def shiftstretchplot(self):

        pngfiles = []
        for file in os.listdir(self.detrender.directoryname):
            if file.endswith('.png'):
                pngfiles.append(file)

        # check to see if this figure already exists in the directroy - it's wavelength bin independent
        if self.inputs.nightname+'_figure_stretchshift.png' in pngfiles: 
            self.speak('strech and shift figure already exists in the detrender directory')
            return

        self.speak('making stretch and shift figure')
        plt.figure(figsize=(16,12))
        gs = plt.matplotlib.gridspec.GridSpec(1, 2, hspace=0.16, wspace=0.02, left=0.05,right=0.99, bottom=0.05, top=0.94)
        fitplots = {}
        colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'w']
        titles = ['stretch', 'shift']
        params = [self.cube.subcube['stretch'], self.cube.subcube['shift']]
        stars = np.hstack((self.inputs.target, self.inputs.comparison))
        pxs = np.hstack((self.inputs.targetpx, self.inputs.comparisonpx))

        for i in range(len(params)):
            if i == 0: fitplots.setdefault(i, []).append(plt.subplot(gs[0,i]))
            else: fitplots.setdefault(i, []).append(plt.subplot(gs[0,i], sharey=fitplots[0][0]))
            for j in range(len(stars)):
                fitplots[i][0].plot(params[i][stars[j]][pxs[j]], range(len(self.cube.subcube['bjd'])), lw=2, alpha=0.5, color=colors[j], label=str(stars[j]))
                fitplots[i][0].plot(params[i][stars[j]][pxs[j]], range(len(self.cube.subcube['bjd'])), 'o', alpha=0.5, color=colors[j])

                fitplots[i][0].axhline(np.where(self.cube.subcube['bjd'] > self.inputs.t0-(1.5*self.inputs.Tdur))[0][0], color='k', ls='-.', alpha = 0.5)
                fitplots[i][0].axhline(np.where(self.cube.subcube['bjd'] < self.inputs.t0+(1.5*self.inputs.Tdur))[0][-1], color='k', ls='-.', alpha = 0.5)

                #plt.xlim(-2, 2)
                plt.ylim(0, len(self.cube.subcube['bjd'])-1)
                plt.xlabel('coeff')
                plt.title(titles[i])
                plt.legend(loc='best')

                if i==0: 
                    plt.ylabel('exposure', fontsize=15)
                    plt.tick_params(axis='y', labelsize=15)
                else: plt.setp(fitplots[i][0].get_yticklabels(), visible=False)


        plt.savefig(self.inputs.saveas+'_figure_stretchshift.png')
        plt.clf()
        plt.close()

    def lmplots(self, wavebin):

        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        self.speak('making lmfit figures')

        self.speak('making normalized rawcounts vs wavelength for target and comparisons figure')
        plt.figure(figsize=(15, 15))
        targinds = np.where(self.wavebin['bininds'][0,0,:])
        targcounts = (self.cube.subcube['raw_counts'][self.inputs.target][self.inputs.targetpx][0]*self.wavebin['bininds'][0,0])[targinds]
        plt.plot(self.cube.subcube['wavelengths'][self.inputs.target][self.inputs.targetpx][0][targinds], targcounts/np.median(targcounts), alpha=0.75, lw=2, label=self.inputs.target)
        for i in range(len(self.inputs.comparison)):
            compinds = np.where(self.wavebin['bininds'][0,i+1])
            compcounts = (self.cube.subcube['raw_counts'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]][0]*self.wavebin['bininds'][0,i+1])[compinds]
            plt.plot(self.cube.subcube['wavelengths'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]][0][compinds], compcounts/np.median(compcounts), alpha=0.75, lw=2, label=self.inputs.comparison[i])
        plt.legend(loc='best')
        plt.xlabel('wavelength [angstroms]', fontsize=20)
        plt.ylabel('normalized raw flux', fontsize=20)
        plt.xlim(self.wavebin['wavelims'][0], self.wavebin['wavelims'][1])
        plt.title(self.inputs.nightname+', '+self.wavefile+' angstroms')
        plt.tight_layout()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_wavebinnedspectrum.png')
        plt.clf()
        plt.close()

        self.speak('making model to offset bjd times')
        #lcbinned = self.targcomp_binned.binned_lcs_dict[self.keys[k]]
        modelobj = ModelMaker(self.inputs, self.wavebin, self.wavebin['lmfit'])
        model = modelobj.makemodel()

        self.speak('making normalized wavelength binned raw counts vs time for target and comparisons figure')
        binnedtarg = np.sum(self.cube.subcube['raw_counts'][self.inputs.target][self.inputs.targetpx] * self.wavebin['bininds'][:,0], 1)[self.wavebin['binnedok']]
        binnedcomp = np.array([np.sum(self.cube.subcube['raw_counts'][self.inputs.comparison[i]][self.inputs.comparisonpx[i]] * self.wavebin['bininds'][:,i+1], 1)[self.wavebin['binnedok']] for i in range(len(self.inputs.comparison))])
        plt.figure()
        plt.plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, binnedtarg/np.median(binnedtarg), '.', alpha=0.5, label=self.inputs.target)
        for i,c in enumerate(binnedcomp):
            plt.plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, c/np.median(c), '.', alpha=0.5, label=self.inputs.comparison[i])
        plt.legend(loc='best')
        plt.xlabel('bjd', fontsize=20)
        plt.ylabel('normalized flux', fontsize=20)
        plt.tight_layout()
        plt.title(self.inputs.nightname+', '+self.wavefile+' angstroms')
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_wavebinnedtimeseries.png')
        plt.clf()
        plt.close()

        self.speak('making normalized wavelength binned raw counts vs time for target and supercomparison figure')
        plt.figure()
        binnedsupercomp = np.sum(binnedcomp, 0)
        plt.plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, binnedtarg/np.median(binnedtarg), '.', alpha=0.5, label='targ')
        plt.plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, binnedsupercomp/np.median(binnedsupercomp), '.', alpha=0.5, label='supercomp')
        plt.legend(loc='best')
        plt.xlabel('bjd', fontsize=20)
        plt.ylabel('normalized nlux', fontsize=20)
        plt.title(self.inputs.nightname+', '+self.wavefile+' angstroms')
        plt.tight_layout()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_wavebinnedtimeseries_supercomp.png')
        plt.clf()
        plt.close()

        self.speak('making lightcurve and lmfit model vs time figure')
        plt.figure(figsize=(14,12))
        gs = plt.matplotlib.gridspec.GridSpec(3, 1, hspace=0.15, wspace=0.15, left=0.08,right=0.98, bottom=0.07, top=0.92)
        lcplots = {}
        lcplots.setdefault('lcplusmodel', []).append(plt.subplot(gs[0:2,0]))
        lcplots.setdefault('residuals', []).append(plt.subplot(gs[2,0]))

        lcplots['lcplusmodel'][0].plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, self.wavebin['lc'], 'k.', alpha=0.7)
        lcplots['lcplusmodel'][0].plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, model, 'bo', alpha=0.5)
        lcplots['lcplusmodel'][0].set_ylabel('lightcurve + model', fontsize=20)

        lcplots['residuals'][0].plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, self.wavebin['lc']-model, 'k.', alpha=0.7)
        lcplots['residuals'][0].set_xlabel('bjd-'+str(self.inputs.t0), fontsize=20)
        lcplots['residuals'][0].set_ylabel('residuals', fontsize=20)

        plt.suptitle(self.inputs.nightname+' lightcurve plus lmfit model, '+self.wavefile+' angstroms')
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_lmfitlcplusmodel.png')
        plt.clf()
        plt.close()

        self.speak('making lmfit detrended lightcurve with batman model vs time figure')
        plt.figure()
        plt.plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, self.wavebin['lc']/modelobj.fitmodel, 'ko', alpha=0.5)
        plt.plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, modelobj.batmanmodel, color="#4682b4", lw=2)
        plt.xlabel('bjd-'+str(self.inputs.t0), fontsize=20)
        plt.ylabel('normalized flux', fontsize=20)
        plt.title('lmfit for '+self.inputs.nightname+', '+self.wavefile+' angstroms', fontsize=20)
        plt.tight_layout()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_lmfitdetrendedlc.png')
        plt.clf()
        plt.close()

        print('making rms vs binsize figure after lmfit')
        time = self.wavebin['compcube']['bjd'] - self.inputs.t0     # days
        time = time*24.*60.                                         # time now in minutes
        sigma_resid = np.std(self.wavebin['lc']-model)
        numbins = 1
        bins = []
        rms = []
        gaussianrms = []
        for i in range(len(model)):
            hist = np.histogram(time, numbins)
            ind_bins, time_bins = hist[0], hist[1]       # number of points in each bin (also helps index), bin limits in units of time [days]
            dtime = time_bins[1]-time_bins[0]
            if dtime < (0.5*self.inputs.Tdur*24.*60.):
                indlow = 0
                num = 0
                for i in ind_bins:
                    num += np.power((np.mean(self.wavebin['lc'][indlow:indlow+i] - model[indlow:indlow+i])), 2.)
                    indlow += i
                calc_rms = np.sqrt(num/numbins)
                if np.isfinite(calc_rms) != True: 
                    numbins +=1 
                    continue
                rms.append(calc_rms)
                bins.append(dtime)    # bin size in units of days
                gaussianrms.append(sigma_resid/np.sqrt(np.mean(ind_bins)))
            numbins += 1
        plt.loglog(np.array(bins), gaussianrms, 'r', lw=2, label='std. err.')
        plt.loglog(np.array(bins), rms, 'k', lw=2, label='rms')
        plt.xlim(bins[-1], bins[0])
        plt.xlabel('bins [minutes]')
        plt.ylabel('rms')
        plt.title('rms vs binsize for wavelengths {0}'.format(self.wavefile))
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_rmsbinsize.png')
        plt.clf()
        plt.close()

        plt.close('all')

    def mcplots(self, wavebin):

        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        self.speak('making mcfit figures')

        self.speak('making walkers vs steps figure')
        paramuncs = np.sqrt(np.diagonal(self.wavebin['linfit'].covar))
        trn_ind = len(self.inputs.tranparams)
        fig, axes = plt.subplots(len(self.inputs.paramlabels), 1, sharex=True, figsize=(16, 12))
        for i in range(len(self.inputs.paramlabels)):
            axes[i].plot(self.wavebin['mcchain'][:, :, i].T, color="k", alpha=0.4)
            axes[i].yaxis.set_major_locator(MaxNLocator(5))
            axes[i].axhline(self.wavebin['mcfit'][i][0], color="#4682b4", lw=2)
            if self.inputs.parambounds[0][i] == True:
                axes[i].axhline(self.wavebin['mcfit'][i][0]-paramuncs[i], color="#4682b4", lw=2, ls='--')
            else:
                axes[i].axhline(self.inputs.parambounds[0][i], color="#4682b4", lw=2, ls='--')
            if self.inputs.parambounds[1][i] == True:
                axes[i].axhline(self.wavebin['mcfit'][i][0]+paramuncs[i], color="#4682b4", lw=2, ls='--')
            else:
                axes[i].axhline(self.inputs.parambounds[1][i], color="#4682b4", lw=2, ls='--')
            axes[i].axvline(self.inputs.burnin, color='k', ls='--', lw=1, alpha=0.4)
            axes[i].set_ylabel(self.inputs.paramlabels[i])
            if i == len(self.inputs.paramlabels)-1: axes[i].set_xlabel("step number")
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcmcchains.png')
        plt.clf()
        plt.close()

        modelobj = ModelMaker(self.inputs, self.wavebin, self.wavebin['mcfit'][:,0])
        model = modelobj.makemodel()

        self.speak('making mcmc corner plot')
        samples = self.wavebin['mcchain'][:,self.inputs.burnin:,:].reshape((-1, len(self.inputs.paramlabels)))
        fig = corner.corner(samples, labels=self.inputs.paramlabels, truths=self.wavebin['lmfit'])
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcmccorner.png')
        plt.clf()
        plt.close()

        self.speak('making lightcurve and mcfit model vs time figure')
        plt.figure(figsize=(14,12))
        gs = plt.matplotlib.gridspec.GridSpec(3, 1, hspace=0.15, wspace=0.15, left=0.08,right=0.98, bottom=0.07, top=0.92)
        lcplots = {}
        lcplots.setdefault('lcplusmodel', []).append(plt.subplot(gs[0:2,0]))
        lcplots.setdefault('residuals', []).append(plt.subplot(gs[2,0]))

        lcplots['lcplusmodel'][0].plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, self.wavebin['lc'], 'k.', alpha=0.7)
        lcplots['lcplusmodel'][0].plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, model, 'bo', alpha=0.5)
        lcplots['lcplusmodel'][0].set_ylabel('lightcurve + model', fontsize=20)

        lcplots['residuals'][0].plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, self.wavebin['lc']-model, 'k.', alpha=0.7)
        lcplots['residuals'][0].set_xlabel('bjd-'+str(self.inputs.t0), fontsize=20)
        lcplots['residuals'][0].set_ylabel('residuals', fontsize=20)

        plt.suptitle(self.inputs.nightname+' lightcurve plus mcmc model, '+self.wavefile+' angstroms', fontsize=20)
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcfitlcplusmodel.png')
        plt.clf()
        plt.close()

        self.speak('making mcfit detrended lightcurve with batman model vs time figure')
        plt.figure()
        plt.plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, self.wavebin['lc']/modelobj.fitmodel, 'ko', alpha=0.5)
        plt.plot(self.cube.subcube['bjd'][self.wavebin['binnedok']]-self.inputs.t0, modelobj.batmanmodel, color="#4682b4", lw=2)
        plt.xlabel('bjd-'+str(self.inputs.t0), fontsize=20)
        plt.ylabel('normalized flux', fontsize=20)
        plt.title('mcfit for '+self.inputs.nightname+', '+self.wavefile+' angstroms')
        plt.tight_layout()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcfitdetrendedlc.png')
        plt.clf()
        plt.close()

        self.speak('making rms vs binsize figure after mcfit')
        time = self.wavebin['compcube']['bjd'] - self.inputs.t0     # days
        time = time*24.*60.                                         # time now in minutes
        sigma_resid = np.std(self.wavebin['lc']-model)
        numbins = 1
        bins = []
        rms = []
        gaussianrms = []
        for i in range(len(model)):
            hist = np.histogram(time, numbins)
            ind_bins, time_bins = hist[0], hist[1]       # number of points in each bin (also helps index), bin limits in units of time [days]
            dtime = time_bins[1]-time_bins[0]
            if dtime < (0.5*self.inputs.Tdur*24.*60.):
                indlow = 0
                num = 0
                for i in ind_bins:
                    num += np.power((np.mean(self.wavebin['lc'][indlow:indlow+i] - model[indlow:indlow+i])), 2.)
                    indlow += i
                calc_rms = np.sqrt(num/numbins)
                if np.isfinite(calc_rms) != True: 
                    numbins +=1 
                    continue
                rms.append(calc_rms)
                bins.append(dtime)    # bin size in units of days
                gaussianrms.append(sigma_resid/np.sqrt(np.mean(ind_bins)))
            numbins += 1
        plt.loglog(np.array(bins), gaussianrms, 'r', lw=2, label='std. err.')
        plt.loglog(np.array(bins), rms, 'k', lw=2, label='rms')
        plt.xlim(bins[-1], bins[0])
        plt.xlabel('bins [minutes]')
        plt.ylabel('rms')
        plt.title('rms vs binsize for wavelengths {0}'.format(self.wavefile))
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_rmsbinsize.png')
        plt.clf()
        plt.close()
