import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np
import os
from ModelMaker import ModelMaker
from ModelMakerJoint import ModelMakerJoint
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator
import corner
from dynesty import plotting as dyplot


class PlotterJoint(Talker):

    '''this class will plot all the things you wish to see'''

    def __init__(self, inputs, cube):
        ''' initialize the plotter 
        directorypath is optional - it will be used by figures.py after detrender is finished '''
        Talker.__init__(self)

        self.inputs = inputs
        self.cube = cube

    def lmplots(self, wavebin, linfits):

        nightcolor = ['tomato', 'orange', 'lawngreen', 'aqua', 'fuchsia']
        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        self.speak('making lmfit figures')

        self.speak('making model to offset bjd times')
        #lcbinned = self.targcomp_binned.binned_lcs_dict[self.keys[k]]
        modelobj = ModelMakerJoint(self.inputs, self.wavebin, self.wavebin['lmfit']['values'])
        models = modelobj.makemodel()

        t0 = []
        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                dtind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                t0.append(self.inputs.toff[n] + self.wavebin['lmfit']['values'][dtind])
            else:
                dtind = int(np.where(np.array(self.inputs.tranlabels[n]) == 'dt')[0])
                t0.append(self.inputs.toff[n] + self.inputs.tranparams[n][dtind])

        self.speak('making lmfit detrended lightcurve with batman model vs time figure')
        plt.figure(figsize=(8, 15))
        gs = plt.matplotlib.gridspec.GridSpec(3, 1, hspace=0.05, wspace=0.0, left=0.14,right=0.98, bottom=0.05, top=0.98)
        plots = {}
        plots['raw'] = plt.subplot(gs[0,0])
        plots['detrend'] = plt.subplot(gs[1,0])
        plots['residual'] = plt.subplot(gs[2,0])
        for n, night in enumerate(self.inputs.nightname):
            xaxis = np.array(self.cube.subcube[n]['bjd'][wavebin['binnedok'][n]]-t0[n])*24*60
            plots['raw'].plot(xaxis, self.wavebin['lc'][n], 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.75, label=str(n+1))
            plots['detrend'].plot(xaxis, self.wavebin['lc'][n]/modelobj.fitmodel[n], 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.6)
            plots['residual'].plot(xaxis, (self.wavebin['lc'][n]-models[n])*1e6, 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.6)
        for n, night in enumerate(self.inputs.nightname):
            xaxis = np.array(self.cube.subcube[n]['bjd'][wavebin['binnedok'][n]]-t0[n])*24*60
            plots['raw'].plot(xaxis, models[n], 'k-', lw=2, alpha=0.4)
            plots['detrend'].plot(xaxis, modelobj.batmanmodel[n], 'k-', lw=2, alpha=0.7)
            plots['residual'].axhline(0.0, .09, .94, color='k', ls='-', lw=2, alpha=0.7)
        plots['raw'].set_ylabel('normalized flux', fontsize=16)
        plots['raw'].tick_params(axis='both', labelsize=12)
        plots['raw'].legend(loc=1, ncol=len(self.inputs.nightname), mode='expand', frameon=False, fontsize=14)
        plots['raw'].set_xticks([])
        plots['detrend'].set_ylabel('normalized flux', fontsize=16)
        plots['detrend'].tick_params(axis='both', labelsize=12)
        plots['detrend'].set_xticks([])
        plots['residual'].tick_params(axis='both', labelsize=12)
        plots['residual'].set_ylabel('residuals [ppm]', fontsize=16)
        plots['residual'].set_xlabel('time from mid-transit [min]', fontsize=16)
        #plt.title('lmfit for joint fit, '+self.wavefile+' angstroms', fontsize=20)
        #plt.tight_layout()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_lmfitdetrendedlc.png')
        plt.clf()
        plt.close()

        
        self.speak('making joint fit residual histogram figure')
        dist = []
        for n, night in enumerate(self.inputs.nightname):
            resid = self.wavebin['lc'][n] - models[n]
            data_unc = np.std(resid)
            dist.append((self.wavebin['lc'][n] - models[n])/data_unc)
        dist = np.hstack(dist)
        n, bins, patches = plt.hist(dist, bins=25, normed=1, color='b', alpha=0.6, label='residuals')
        gaussiandist = np.random.randn(10000)
        ngauss, binsgauss, patchesgauss = plt.hist(gaussiandist, bins=25, normed=1, color='r', alpha=0.6, label='gaussian')
        plt.title('residuals for '+self.wavefile, fontsize=20)
        plt.xlabel('uncertainty-weighted residuals', fontsize=20)
        plt.ylabel('number of data points', fontsize=20)
        plt.legend()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_residuals_hist.png')
        plt.clf()
        plt.close()

        self.speak('making lmfit residuals plot for each lmfit call')
        for i, linfit in enumerate(linfits):
            plt.plot(range(len(linfit.residual)), linfit.residual, '.', alpha=.6, label='linfit'+str(i)+', chi2 = '+str(linfit.chisqr))
        plt.xlabel('data number')
        plt.ylabel('residuals')
        plt.legend(loc='best')
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_linfit_residuals.png')
        plt.clf()
        plt.close()
        

        plt.close('all')

    def mcplots(self, wavebin):

        nightcolor = ['tomato', 'orange', 'lawngreen', 'aqua', 'fuchsia']
        self.wavebin = wavebin
        self.wavefile = str(self.wavebin['wavelims'][0])+'-'+str(self.wavebin['wavelims'][1])

        self.speak('making mcfit figures')

        if self.inputs.mcmccode == 'emcee':        

            self.speak('making walkers vs steps figure')
            fig, axes = plt.subplots(len(self.inputs.freeparamnames), 1, sharex=True, figsize=(16, 12))
            for i, name in enumerate(self.inputs.freeparamnames):
                axes[i].plot(self.wavebin['mcfit']['chain'][:, :, i].T, color="k", alpha=0.4)
                axes[i].yaxis.set_major_locator(MaxNLocator(5))
                axes[i].axhline(self.wavebin['mcfit']['values'][i][0], color="#4682b4", lw=2)
                if self.inputs.freeparambounds[0][i] == True:
                    axes[i].axhline(self.wavebin['mcfit']['values'][i][0]-self.wavebin['lmfit']['uncs'][i], color="#4682b4", lw=2, ls='--')
                else:
                    axes[i].axhline(self.inputs.freeparambounds[0][i], color="#4682b4", lw=2, ls='--')
                if self.inputs.freeparambounds[1][i] == True:
                    axes[i].axhline(self.wavebin['mcfit']['values'][i][0]+self.wavebin['lmfit']['uncs'][i], color="#4682b4", lw=2, ls='--')
                else:
                    axes[i].axhline(self.inputs.freeparambounds[1][i], color="#4682b4", lw=2, ls='--')
                axes[i].axvline(self.inputs.burnin, color='k', ls='--', lw=1, alpha=0.4)
                axes[i].set_ylabel(self.inputs.freeparamnames[i])
                if i == len(self.inputs.freeparamnames)-1: axes[i].set_xlabel("step number")
            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcmcchains.png')
            plt.clf()
            plt.close()

            
            self.speak('making mcmc corner plot')
            samples = self.wavebin['mcfit']['chain'][:,self.inputs.burnin:,:].reshape((-1, len(self.inputs.freeparamnames)))
            fig = corner.corner(samples, labels=self.inputs.freeparamnames, truths=self.wavebin['lmfit']['values'])
            plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcmccorner.png')
            plt.clf()
            plt.close()

        elif self.inputs.mcmccode == 'dynesty':

            truths = self.wavebin['lmfit']['values']
            # add the u0 and u1 truths, as well as the s parameter
            v0, v1 = wavebin['ldparams'][0][0], wavebin['ldparams'][0][1]
            truths.append(v0)
            truths.append(v1)
            for n in range(len(self.inputs.nightname)): truths.append(1)

            # trace plot
            fig, axes = dyplot.traceplot(self.wavebin['mcfit']['results'], labels=self.inputs.freeparamnames, post_color='royalblue', truths=truths, truth_color='firebrick', truth_kwargs={'alpha': 0.8}, fig=plt.subplots(len(self.inputs.freeparamnames), 2, figsize=(12, 30)), trace_kwargs={'edgecolor':'none'})
            plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcmcchains.png')
            plt.clf()
            plt.close()

            '''
            # corner plot
            fig, axes = dyplot.cornerplot(self.wavebin['mcfit']['results'], labels=self.inputs.freeparamnames, truths=truths, show_titles=True, title_kwargs={'y': 1.04}, fig=plt.subplots(len(self.inputs.freeparamnames), len(self.inputs.freeparamnames), figsize=(15, 15)))
            plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcmccorner.png')
            plt.clf()
            plt.close()
            '''
            

        t0 = []
        for n, night in enumerate(self.inputs.nightname):
            if 'dt'+str(n) in self.inputs.freeparamnames:
                dtind = int(np.where(np.array(self.inputs.freeparamnames) == 'dt'+str(n))[0])
                t0.append(self.inputs.toff[n] + self.wavebin['mcfit']['values'][dtind][0])
            else:
                dtind = int(np.where(np.array(self.inputs.tranlabels[n]) == 'dt')[0])
                t0.append(self.inputs.toff[n] + self.inputs.tranparams[n][dtind])

        modelobj = ModelMakerJoint(self.inputs, self.wavebin, self.wavebin['mcfit']['values'][:,0])
        models = modelobj.makemodel()

        self.speak('making mcfit detrended lightcurve with batman model vs time figure')

        plt.figure(figsize=(8, 15))
        gs = plt.matplotlib.gridspec.GridSpec(3, 1, hspace=0.05, wspace=0.0, left=0.14,right=0.98, bottom=0.05, top=0.98)
        plots = {}
        plots['raw'] = plt.subplot(gs[0,0])
        plots['detrend'] = plt.subplot(gs[1,0])
        plots['residual'] = plt.subplot(gs[2,0])
        for n, night in enumerate(self.inputs.nightname):
            xaxis = np.array(self.cube.subcube[n]['bjd'][wavebin['binnedok'][n]]-t0[n])*24*60
            plots['raw'].plot(xaxis, self.wavebin['lc'][n], 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.75, label=str(n+1))
            plots['detrend'].plot(xaxis, self.wavebin['lc'][n]/modelobj.fitmodel[n], 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.6)
            plots['residual'].plot(xaxis, (self.wavebin['lc'][n]-models[n])*1e6, 'o', color=nightcolor[n], markeredgecolor=nightcolor[n], alpha=0.6)
        for n, night in enumerate(self.inputs.nightname):
            xaxis = np.array(self.cube.subcube[n]['bjd'][wavebin['binnedok'][n]]-t0[n])*24*60
            plots['raw'].plot(xaxis, models[n], 'k-', lw=2, alpha=0.4)
            plots['detrend'].plot(xaxis, modelobj.batmanmodel[n], 'k-', lw=2, alpha=0.7)
            plots['residual'].axhline(0.0, .09, .94, color='k', ls='-', lw=2, alpha=0.7)
        plots['raw'].set_ylabel('normalized flux', fontsize=16)
        plots['raw'].tick_params(axis='both', labelsize=12)
        plots['raw'].legend(loc=1, ncol=len(self.inputs.nightname), mode='expand', frameon=False, fontsize=14)
        plots['raw'].set_xticks([])
        plots['detrend'].set_ylabel('normalized flux', fontsize=16)
        plots['detrend'].tick_params(axis='both', labelsize=12)
        plots['detrend'].set_xticks([])
        plots['residual'].tick_params(axis='both', labelsize=12)
        plots['residual'].set_ylabel('residuals [ppm]', fontsize=16)
        plots['residual'].set_xlabel('time from mid-transit [min]', fontsize=16)
        #plt.title('mcfit for joint fit, '+self.wavefile+' angstroms', fontsize=20)
        #plt.tight_layout()
        plt.savefig(self.inputs.saveas+'_'+self.wavefile+'_figure_mcfitdetrendedlc.png')
        plt.clf()
        plt.close()


