import sys
sys.path.append('/home/hdiamond/local/lib/python2.7/site-packages/')
sys.path.append('/h/mulan0/code/')
sys.path.append('/h/mulan0/code/mosasaurus')
sys.path.append('/h/mulan0/code/detrendersaurus')
import numpy as np
import matplotlib.pyplot as plt
import astrotools.modeldepths as md
import astropy.units as u
from ModelMaker import ModelMaker
from Inputs import Inputs
import collections
import os
import datetime

# this code will take the path to a folder that includes directories of outputs form the detrender
# the output will be a figure of the lightcurves and the transmission spectrum (if flagged)
# if there are many lightcurves in one folder, these will be stacked vertically
# if there are multiple nights, they will be stacked horizontally
# you may need to change the basepath variable to suit your directory setup

# where the analysis files are stored
basepath = '/h/mulan0/analysis/GJ1132/alldata/'
directoryname = 'BICs'

# get model transmission spectrum
specpath = '/home/hdiamond/local/lib/Exo_Transmit/Spectra/'
specoutfile1 = 'GJ1132b_H2.dat'
specdatafile1 = specpath+specoutfile1

#transmission constants
bin_size = 0.0200          # microns
wave_range = [.7, 1.03]   # microns
smooth = [51, 5]
scale = .8
otherpplsdata = False
colors = ['royalblue', 'firebrick', 'goldenrod', 'forestgreen', 'purple']

H2 = md.model_spectrum(specdatafile1, bin_size, wave_range, smooth)
modelspectra = [H2]

subdirectories = os.listdir(basepath+directoryname)
loldict = {}

def loadrun(directory, inputs, wavefile):
    wavebin = np.load(directory+inputs.nightname+'_'+wavefile+'.npy')[()]

    modelobj = ModelMaker(inputs, wavebin, wavebin['lmfit'])
    model = modelobj.makemodel()

    rprsind = int(np.where(np.array(inputs.paramlabels) == 'rp')[0][0])
    try:
        rprs = wavebin['mcfit'][rprsind][0]
        rprsunc = np.mean([wavebin['mcfit'][rprsind][1], wavebin['mcfit'][rprsind][2]])
    except(KeyError):
        rprs = wavebin['lmfit'][rprsind]
        rprsunc = np.sqrt(np.diagonal(wavebin['linfit'].covar))[rprsind]
    depth = rprs**2
    depthunc = depth*2*(rprsunc/rprs)

    loldict[inputs.nightname][w]['lc'] = wavebin['lc']
    loldict[inputs.nightname][w]['binnedok'] = wavebin['binnedok']
    loldict[inputs.nightname][w]['fitmodel'] = modelobj.fitmodel
    loldict[inputs.nightname][w]['batmanmodel'] = modelobj.batmanmodel
    loldict[inputs.nightname][w]['depth'] = depth
    loldict[inputs.nightname][w]['depthunc'] = depthunc


for dir in subdirectories:
    path = basepath+directoryname+'/'+dir+'/'
    print 'reading inputs from ', dir
    inputs = Inputs(path)
    print 'loading cube from', dir
    subcube = np.load(path+inputs.nightname+'_subcube.npy')[()]
    loldict[inputs.nightname] = collections.defaultdict(dict)
    loldict[inputs.nightname]['bjd'] = subcube['bjd']
    loldict[inputs.nightname]['t0'] = inputs.t0
    loldict[inputs.nightname]['wavelims'] = inputs.wavelength_lims
    numbins = np.floor((inputs.wavelength_lims[1] - inputs.wavelength_lims[0])/inputs.binlen)
    binlen = (inputs.wavelength_lims[1] - inputs.wavelength_lims[0])/numbins
    loldict[inputs.nightname]['binlen'] = binlen
    print 'working on night', inputs.nightname

    print 'reading wavefiles...'
    wavefiles = subcube['wavebin']['wavefiles']
    loldict[inputs.nightname]['wavefiles'] = wavefiles

    for w in wavefiles:
        print '    wavelength bin', w
        loadrun(path, inputs, w)


numnights = len(subdirectories)
nights = loldict.keys()
nights = sorted(nights, key=lambda x: datetime.datetime.strptime(x[:-3], '%Y_%m_%d'))
        

##################################################################################
############################ plot lightcurves ####################################
##################################################################################
def lightcurves():
    plt.figure(figsize=(20, 12))
    gs = plt.matplotlib.gridspec.GridSpec(1, len(subdirectories), hspace=0.16, wspace=0.02, left=0.06,right=0.99, bottom=0.04, top=0.94)
    lcplots = {}
    offset = 0.01

    for i, night in enumerate(nights):
        if i == 0: lcplots.setdefault(i, []).append(plt.subplot(gs[0,i]))
        else: lcplots.setdefault(i, []).append(plt.subplot(gs[0,i], sharey=lcplots[0][0]))
        wavefiles = loldict[night]['wavefiles']
        for w, wave in enumerate(wavefiles):

            binnedok = loldict[night][wave]['binnedok']
            t0 = loldict[night]['t0']
            bjdbinned = loldict[night]['bjd'][binnedok] - t0
            resid = loldict[night][wave]['lc'] - loldict[night][wave]['fitmodel']*loldict[night][wave]['batmanmodel']
            rms = np.std(resid)
            

            # need to deal with self.t0 changing during run; this is a secondary issue as you will likely only be plotting from runs where you have fixed self.t0

            lcplots[i][0].plot(bjdbinned, (loldict[night][wave]['lc']/loldict[night][wave]['fitmodel'])-offset*w, 'ko', alpha=0.5)
            lcplots[i][0].plot(bjdbinned, (loldict[night][wave]['batmanmodel'])-offset*w, color=colors[i], lw=4, alpha=0.8)
            #lcplots[i][0].set_xlim(-0.05, 0.05)
            lcplots[i][0].text(-0.048, 1.0025-offset*w, wave+' A', color='#ff471a', weight='bold', fontsize=12)
            lcplots[i][0].text(0.012, 1.0025-offset*w, 'RMS: {0:.0f} ppm'.format(rms*1e6), color='#ff471a', weight='bold', fontsize=12)
            #lcplots[i][0].text(0.0125, 1.0025-offset*k, 'x exp. noise: {0:.2f}'.format(mcmcRMS/expnoise), color='#ff471a', weight='bold', fontsize=12)

            plt.xlabel('bjd-'+str(t0), fontsize=15)
            plt.xlim(-.05, .05)
            if i==0: 
                plt.ylabel('normalized flux', fontsize=15)
                plt.tick_params(axis='y', labelsize=15)
            else: plt.setp(lcplots[i][0].get_yticklabels(), visible=False)
    #plt.ylim(0.9955, 1.002)
    plt.ylim(.84, 1.006)
    plt.suptitle('wavelength range: ' + str(loldict[night]['wavelims']) + ' A, binsize: ' + str(loldict[night]['binlen']) + ' A', fontsize=15)
    plt.show()

##################################################################################
############################ plot transmission ####################################
##################################################################################
def transmission(flag='absolute', model=True, medavg=True):
    plt.figure(figsize=(14, 8))

    if flag == 'absolute':

        if model:
            for m in modelspectra:
                model_wavelengths, model_depths, model_binned_wavelengths, model_binned_depths = np.array(m[0]), np.array(m[1]), np.array(m[2]), np.array(m[3])
                plt.plot(model_wavelengths, model_depths*scale*100., color='k', alpha=0.5, linewidth=2, label=r'$100\%\ H_2/He$')
                plt.plot(model_wavelengths, [.215 for i in model_wavelengths], color='k', ls='--', alpha=0.5, linewidth=2, label=r'$\mathrm{flat}$')
                #plt.plot(model_binned_wavelengths, model_binned_depths*scale*100., 'ks', alpha=0.75)

        if medavg:
            alldepth = []
            alldepthunc = []
        for i, night in enumerate(nights):
            wavefiles = loldict[night]['wavefiles']
            wavefilelims = [[float(wave) for wave in wavefile.split('-')] for wavefile in wavefiles]

            xaxis = [np.mean(wavefilelim)/10000. for wavefilelim in wavefilelims]
            depth =  [loldict[night][w]['depth']*100. for w in wavefiles]
            depthunc =  [loldict[night][w]['depthunc']*100. for w in wavefiles]
            if medavg:
                alldepth.append(depth)
                alldepthunc.append(depthunc)
                plt.errorbar(xaxis, depth, yerr=depthunc, fmt='o', markersize=8, color=colors[i], markeredgecolor=colors[i], ecolor=colors[i], elinewidth=2, capsize=0, alpha=0.25, label=night)
            else: plt.errorbar(xaxis, depth, yerr=depthunc, fmt='o', markersize=8, color=colors[i], markeredgecolor=colors[i], ecolor=colors[i], elinewidth=2, capsize=0, alpha=0.7, label=night)

        if medavg: 
            alldepth = np.array(alldepth)
            alldepthunc = np.array(alldepthunc)
            weighteddepth = np.sum(alldepth/alldepthunc**2, 0)/np.sum(1./alldepthunc**2, 0)
            weighteddepthunc = np.sqrt(1./np.sum(1./alldepthunc**2, 0))
            print 'inverse variance weighted depths:'
            print weighteddepth
            try: plt.errorbar(xaxis, weighteddepth, yerr=weighteddepthunc, fmt='o', markersize=8, color='k', markeredgecolor='k', ecolor='k', elinewidth=2, capsize=0, alpha=0.9, label='weighted average')
            except(ValueError):
                print 'check that all of your directories have the same number of wavelength bins!'
                print 'cannot take the inverse variance weighted average of differently binned points'
            

    if flag == 'normalized':
        # calculate median transit depth across all wavelength bins, acros all nights
        allwavefiles = np.array([loldict[night]['wavefiles'] for night in nights])
        meddepth = np.median([[loldict[night][w]['depth'] for w in allwavefiles[n]] for n,night in enumerate(nights)])*100.
        print 'median depth: ', meddepth

        if model:
            for m in modelspectra:
                model_wavelengths, model_depths, model_binned_wavelengths, model_binned_depths = np.array(m[0]), np.array(m[1]), np.array(m[2]), np.array(m[3])
                plt.plot(model_wavelengths, model_depths*scale*100./np.median(model_depths*scale*100.)*meddepth, color='k', alpha=0.5, linewidth=2, label=r'$100\%\ H_2/He$')
                plt.plot(model_wavelengths, [meddepth for i in model_wavelengths], color='k', ls='--', alpha=0.5, linewidth=2, label=r'$\mathrm{flat}$')
                #plt.plot(model_binned_wavelengths, model_binned_depths*scale*100./np.median(model_depths*scale*100.), 'ks', alpha=0.75)

        if medavg:
            alldepth = []
            alldepthunc = []

        for i, night in enumerate(nights):
            wavefiles = loldict[night]['wavefiles']
            wavefilelims = [[float(wave) for wave in wavefile.split('-')] for wavefile in wavefiles]

            xaxis = [np.mean(wavefilelim)/10000. for wavefilelim in wavefilelims]
            depth =  [loldict[night][w]['depth']*100. for w in wavefiles]
            depthunc =  [loldict[night][w]['depthunc']*100. for w in wavefiles]
            if medavg:
                alldepth.append(depth)
                alldepthunc.append(depthunc)
                plt.errorbar(xaxis, depth/np.median(depth)*meddepth, yerr=depthunc/np.median(depth)*meddepth, fmt='o', markersize=8, color=colors[i], markeredgecolor=colors[i], ecolor=colors[i], elinewidth=2, capsize=0, alpha=0.25, label=night)
            else: plt.errorbar(xaxis, depth/np.median(depth)*meddepth, yerr=depthunc/np.median(depth)*meddepth, fmt='o', markersize=8, color=colors[i], markeredgecolor=colors[i], ecolor=colors[i], elinewidth=2, capsize=0, alpha=0.7, label=night)

        if medavg: 
            alldepth = np.array(alldepth)
            alldepthunc = np.array(alldepthunc)
            weighteddepth = np.sum(alldepth/alldepthunc**2, 0)/np.sum(1./alldepthunc**2, 0)
            weighteddepthunc = np.sqrt(1./np.sum(1./alldepthunc**2, 0))
            print 'inverse variance weighted depths:'
            print weighteddepth
            try: plt.errorbar(xaxis, weighteddepth/np.median(weighteddepth)*meddepth, yerr=weighteddepthunc/np.median(weighteddepth)*meddepth, fmt='o', markersize=8, color='k', markeredgecolor='k', ecolor='k', elinewidth=2, capsize=0, alpha=0.9, label='weighted average')
            except(ValueError):
                print 'check that all of your directories have the same number of wavelength bins!'
                print 'cannot take the inverse variance weighted average of differently binned points'

    plt.xlim(wave_range)
    plt.ylabel('transit depth [%]')
    plt.xlabel('wavelength [microns]')
    plt.tight_layout()
    plt.show()

