import sys
sys.path.append('/home/hdiamond/local/lib/python2.7/site-packages/')
sys.path.append('/h/mulan0/code/')
sys.path.append('/h/mulan0/code/mosasaurus')
sys.path.append('/h/mulan0/code/detrendersaurus')
import numpy as np
import matplotlib.pyplot as plt
import astrotools.modeldepths as md
import astropy.units as u
from ModelMakerJoint import ModelMakerJoint
from InputsJoint import InputsJoint
import collections
import os
from datetime import datetime

# this code will take the path to a folder that includes directories of outputs form the detrender
# the output will be a figure of the lightcurves and the transmission spectrum (if flagged)
# if there are many lightcurves in one folder, these will be stacked vertically
# if there are multiple nights, they will be stacked horizontally
# you may need to change the basepath variable to suit your directory setup

# where the analysis files are stored
basepath = '/h/mulan0/analysis/GJ1132/alldata/'
directoryname = 'joint'
#rundir = '2017-09-25-18:37_bin200A-dynesty/'
#rundir = 'bin200A-dynesty-compilation/'
rundir = 'bin200A-dynestynew-compilation/'

# get model transmission spectrum
specpath = '/home/hdiamond/local/lib/Exo_Transmit/Spectra/'
specoutfile1 = 'GJ1132b_H2.dat'
specdatafile1 = specpath+specoutfile1

#transmission constants
bin_size = 0.0200          # microns
wave_range = [.7, 1.03]   # microns
smooth = [51, 5]
scale = 1.0
otherpplsdata = False
colors = ['royalblue', 'firebrick', 'goldenrod', 'forestgreen', 'purple']

H2 = md.model_spectrum(specdatafile1, bin_size, wave_range, smooth)
modelspectra = [H2]

subdirectories = os.listdir(basepath+directoryname)
loldict = {}

def loadrun(directory, inputs, wavefile):
    wavebin = np.load(directory+'joint_'+wavefile+'.npy')[()]

    rprsind = int(np.where(np.array(inputs.freeparamnames) == 'rp0')[0])
    try:
        rprs = wavebin['mcfit']['values'][rprsind][0]
        rprsunc = np.array([wavebin['mcfit']['values'][rprsind][1], wavebin['mcfit']['values'][rprsind][2]])
        modelobj = ModelMakerJoint(inputs, wavebin, wavebin['mcfit']['values'][:,0])
        models = modelobj.makemodel()
        print '        using mcfit values'
    except(KeyError):
        rprs = wavebin['lmfit']['values'][rprsind]
        rprsunc = wavebin['lmfit']['uncs'][rprsind]
        modelobj = ModelMakerJoint(inputs, wavebin, wavebin['lmfit']['values'])
        models = modelobj.makemodel()
        print '        using lmfit values'
    depth = rprs**2
    depthunc = depth*2*(rprsunc/rprs)
    print '        depth [%]', depth*100.
    print '        depth unc [%, %]', depthunc*100.

    resid = []
    for n, night in enumerate(inputs.nightname):
        loldict[night][wavefile]['lc'] = wavebin['lc'][n]
        loldict[night][wavefile]['binnedok'] = wavebin['binnedok'][n]
        loldict[night][wavefile]['fitmodel'] = modelobj.fitmodel[n]
        loldict[night][wavefile]['batmanmodel'] = modelobj.batmanmodel[n]
        resid.append(wavebin['lc'][n] - models[n])
        print '        x expected noise for {0}: {1}'.format(night, np.std(resid[n])/wavebin['photnoiselim'][n])
    print '        median expected noise for joint fit: {0}'.format(np.median([np.std(resid[n])/wavebin['photnoiselim'][n] for n in range(len(inputs.nightname))]))
    loldict['joint'][wavefile]['depth'] = depth
    loldict['joint'][wavefile]['depthunc'] = depthunc


path = basepath+directoryname+'/run/'+rundir
jointdirectories = [d for d in os.listdir(basepath+directoryname) if os.path.isdir(os.path.join(basepath+directoryname, d))]
if 'run' in jointdirectories: jointdirectories.remove('run')
jointdirectories = sorted(jointdirectories, key=lambda x: datetime.strptime(x[:-3], '%Y_%m_%d'))

print 'reading inputs from ', basepath+directoryname
inputs = InputsJoint(jointdirectories, path)
print 'loading joint cube'
subcube = np.load(path+'joint_subcube.npy')[()]
for n, night in enumerate(inputs.nightname):
    loldict[night] = collections.defaultdict(dict)
    loldict[night]['bjd'] = subcube[n]['bjd']
    loldict[night]['t0'] = inputs.t0[n]     # make sure this number is correct - may have to change in input files
loldict['joint'] = collections.defaultdict(dict)
loldict['joint']['wavelims'] = inputs.wavelength_lims
numbins = np.floor((inputs.wavelength_lims[1] - inputs.wavelength_lims[0])/inputs.binlen)
binlen = (inputs.wavelength_lims[1] - inputs.wavelength_lims[0])/numbins
loldict['joint']['binlen'] = binlen

print 'reading wavefiles...'
wavefiles = subcube[0]['wavebin']['wavefiles']
loldict['joint']['wavefiles'] = wavefiles

for w in wavefiles:
    print '    wavelength bin', w
    loadrun(path, inputs, w)


##################################################################################
############################ plot lightcurves ####################################
##################################################################################
def lightcurves():
    plt.figure(figsize=(9, 12))
    gs = plt.matplotlib.gridspec.GridSpec(1, 1, hspace=0.16, wspace=0.02, left=0.1,right=0.99, bottom=0.05, top=0.99)
    lcplots = {}
    offset = 0.01

    lcplots.setdefault(0, []).append(plt.subplot(gs[0,0]))
    #else: lcplots.setdefault(i, []).append(plt.subplot(gs[0,i], sharey=lcplots[0][0]))
    wavefiles = loldict['joint']['wavefiles']


    for w, wave in enumerate(wavefiles):
        resid = []
        for n, night in enumerate(inputs.nightname):
            binnedok = loldict[night][wave]['binnedok']
            t0 = loldict[night]['t0']
            bjdbinned = loldict[night]['bjd'][binnedok] - t0
            resid.append(loldict[night][wave]['lc'] - loldict[night][wave]['fitmodel']*loldict[night][wave]['batmanmodel'])

            # need to deal with self.t0 changing during run; this is a secondary issue as you will likely only be plotting from runs where you have fixed self.t0

            lcplots[0][0].plot(bjdbinned, (loldict[night][wave]['lc']/loldict[night][wave]['fitmodel'])-offset*w, 'ko', alpha=0.5)
            lcplots[0][0].plot(bjdbinned, (loldict[night][wave]['batmanmodel'])-offset*w, color=colors[0], lw=4, alpha=0.8)
            #lcplots[0][0].set_xlim(-0.05, 0.05)
        allresid = np.hstack(resid)
        rms = np.std(allresid)
        lcplots[0][0].text(-0.048, 1.003-offset*w, wave+' A', color='#ff471a', weight='bold', fontsize=14)
        lcplots[0][0].text(0.028, 1.003-offset*w, 'RMS: {0:.0f} ppm'.format(rms*1e6), color='#ff471a', weight='bold', fontsize=14)
        #lcplots[0][0].text(0.0125, 1.0025-offset*k, 'x exp. noise: {0:.2f}'.format(mcmcRMS/expnoise), color='#ff471a', weight='bold', fontsize=12)

    plt.xlabel('time from mid-transit', fontsize=15)
    plt.xlim(-.05, .05)
    plt.ylabel('normalized flux', fontsize=15)
    plt.tick_params(axis='y', labelsize=15)
    #plt.ylim(0.9955, 1.002)
    plt.ylim(.84, 1.006)
    #plt.suptitle('wavelength range: ' + str(loldict['joint']['wavelims']) + ' A, binsize: ' + str(loldict['joint']['binlen']) + ' A', fontsize=15)
    plt.show()

##################################################################################
############################ plot transmission ####################################
##################################################################################
def transmission(flag='absolute', model=True):
    plt.figure(figsize=(14, 8))

    if flag == 'absolute':

        if model:
            for m in modelspectra:
                model_wavelengths, model_depths, model_binned_wavelengths, model_binned_depths = np.array(m[0]), np.array(m[1]), np.array(m[2]), np.array(m[3])
                #plt.plot(model_wavelengths, model_depths*scale*100., color='k', alpha=0.5, linewidth=2, label=r'$100\%\ H_2/He$')
                #plt.plot(model_wavelengths, [np.mean(model_depths*scale*100.) for i in model_wavelengths], color='k', ls='--', alpha=0.5, linewidth=2, label=r'$\mathrm{flat}$')
                #plt.plot(model_binned_wavelengths, model_binned_depths*scale*100., 'ks', alpha=0.75)

            wavefiles = loldict['joint']['wavefiles']
            wavefilelims = [[float(wave) for wave in wavefile.split('-')] for wavefile in wavefiles]

            xaxis = [np.mean(wavefilelim)/10000. for wavefilelim in wavefilelims]
            depth =  [loldict['joint'][w]['depth']*100. for w in wavefiles]
            depthunc =  [np.mean(loldict['joint'][w]['depthunc']*100.) for w in wavefiles]
            plt.errorbar(xaxis, depth, yerr=depthunc, fmt='o', markersize=8, color=colors[0], markeredgecolor=colors[0], ecolor=colors[0], elinewidth=2, capsize=0, alpha=0.8, label=r'$\rm joint\ fit\ data$')

            # linear fit to the transit depths
            z = np.polyfit(xaxis, depth, deg=1)
            p = np.poly1d(z)
            x = np.linspace(wave_range[0], wave_range[1], 100)

            model_binned_depths = []
            fit_binned_depths = []
            for i in range(len(xaxis)):
                bininds = np.where((model_wavelengths >= wavefilelims[i][0]/10000.) & (model_wavelengths <= wavefilelims[i][1]/10000.))[0]
                model_binned_depths.append(np.mean(model_depths[bininds]*scale*100.))
                bininds = np.where((x >= wavefilelims[i][0]/10000.) & (x <= wavefilelims[i][1]/10000.))[0]
                fit_binned_depths.append(np.mean(p(x[bininds])))
            model_binned_depths = np.array(model_binned_depths)
            fit_binned_depths = np.array(fit_binned_depths)

            # figure out how to move the atmosphere model to best match the points (only shifting, not chaning the size of the features)
            chisq_H2 = np.inf
            best_offset = 0.0
            for i in np.linspace(-.1, .1, 1000):
                model_offset = model_binned_depths + i
                chisq_offset = np.sum(((depth - model_offset)/np.mean(depthunc))**2)
                if chisq_offset < chisq_H2: 
                    chisq_H2 = chisq_offset
                    best_offset = i
                #chisq_H2_array.append(chisq_offset)
            #plt.figure()
            #n, bins, patches = plt.hist(chisq_H2_array)
            #plt.show()
        
            model_binned_depths = model_binned_depths + best_offset

            chisq = np.sum(np.power((depth - model_binned_depths)/np.mean(depthunc), 2.))
            print 'chisq_H2:', chisq
            chisq_flat = np.sum(np.power((depth - np.array([np.median(depth) for i in depth]))/np.mean(depthunc), 2.))
            print 'chisq_flat:', chisq_flat
            chisq_fit = np.sum(np.power((depth - fit_binned_depths)/np.mean(depthunc), 2.))
            print 'chisq_fit:', chisq_fit

            plt.figure(1)
            plt.plot(model_wavelengths, model_depths*scale*100.+best_offset, color='k', alpha=0.5, linewidth=2, label=r'$100\%\ H_2/He,\ \chi^2 = {0}$'.format('%.2f'%chisq))
            plt.plot(x, [np.median(depth) for i in x], 'k--', lw=2, alpha=0.6, label=r'$\rm flat\ \chi^2 = {0}$'.format('%.2f'%chisq_flat))
            plt.plot(x, p(x), 'k:', lw=2, alpha=0.6, label=r'$\rm fit\ \chi^2 = {0}$'.format('%.2f'%chisq_fit))

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

    plt.legend(loc=2, ncol=2)
    plt.xlim(wave_range)
    plt.ylabel('transit depth [%]')
    plt.xlabel('wavelength [microns]')
    plt.tight_layout()
    plt.show()

################################################################################################
################# plot all data ################################################################
################################################################################################

def alldata():

    plt.figure(figsize=(14, 8))

    wavefiles = loldict['joint']['wavefiles']
    wavefilelims = [[float(wave) for wave in wavefile.split('-')] for wavefile in wavefiles]

    xaxis = [np.mean(wavefilelim)/10000. for wavefilelim in wavefilelims]
    depth =  [loldict['joint'][w]['depth']*100. for w in wavefiles]
    depthunc =  [np.mean(loldict['joint'][w]['depthunc']*100.) for w in wavefiles]
    plt.errorbar(xaxis, depth, yerr=depthunc, fmt='o', markersize=8, color='royalblue', markeredgecolor='royalblue', ecolor='royalblue', elinewidth=2, capsize=0, alpha=0.8, label=r'This work')

    # southworth data
    #southworth_star = 0.255 * u.solRad

    #southworth_g_wave = 0.477
    #southworth_g = np.array([1.209, 1.475, 1.567, 1.318, 1.221, 1.457, 1.515, 1.570, 1.255]) * u.earthRad
    #southworth_g_uncs = np.array([.154, .091, .151, .122, .12, .228, .146, .121, .183]) * u.earthRad
    #southworth_g_rprs = np.array((southworth_g/southworth_star).decompose())
    #southworth_g_rprs_uncs = np.array((southworth_g_uncs/southworth_star).decompose())
    #southworth_g_depths = southworth_g_rprs**2
    #southworth_g_depths_uncs = southworth_g_depths*np.sqrt(2.*((southworth_g_rprs_uncs/southworth_g_rprs)**2))
    #plt.errorbar([southworth_g_wave for i in range(len(southworth_g))], southworth_g_depths*100., yerr=southworth_g_depths_uncs*100., color='m', fmt='o', markersize=8, markeredgecolor='m', ecolor='m', elinewidth=2, capsize=0, alpha=0.2, label='Southworth')        

    southworth_wave = np.array([.477, .623, .763, .913, 1.23, 1.645, 2.165])
    southworth_rb = np.array([.00382, .00402, .00386, .00446, .00354, .00324, .00473])
    southworth_rb_unc = np.array([.00011, .00009, .00006, .00015, .00045, .00044, .00058])
    southworth_RA_a = .07733476
    southworth_rprs = southworth_rb/southworth_RA_a
    southworth_rprs_uncs = southworth_rb_unc/southworth_RA_a
    southworth_depths = southworth_rprs**2
    southworth_depths_uncs = southworth_depths*2.*(southworth_rprs_uncs/southworth_rprs)
    southworth_labels = ['g', 'r', 'i', 'z', 'J', 'H', 'K']
    plt.errorbar(southworth_wave, southworth_depths*100., yerr=southworth_depths_uncs*100., color='darkorchid', fmt='o', markersize=8, markeredgecolor='darkorchid', ecolor='darkorchid', elinewidth=2, capsize=0, alpha=0.8, label='Southworth, et al. (2017)')
    #plt.plot(southworth_wave, southworth_depths/np.median(weighted_depths), yerr=southworth_depths_uncs/np.median(weighted_depths), color='k', fmt='o', markersize=10, markeredgecolor='k', ecolor=color'k', elinewidth=3, capsize=4, alpha=0.6, label='Southworth')

    # MEarth data
    mearth_wave = 0.9
    mearth_rprs = 0.0455
    mearth_rprs_unc = 0.0006
    mearth_depth = mearth_rprs**2
    mearth_depth_unc = mearth_depth*np.sqrt(2.*((mearth_rprs_unc/mearth_rprs)**2))
    mearth_label = 'MEarth, Dittmann et al. (2017)'
    plt.errorbar(mearth_wave, mearth_depth*100., yerr=mearth_depth_unc*100., color='forestgreen', fmt='o', markersize=8, markeredgecolor='forestgreen', ecolor='forestgreen', elinewidth=2, capsize=0, alpha=0.8, label=mearth_label)
    #plt.errorbar(mearth_wave, mearth_depth/np.median(weighted_depths), yerr=mearth_depth_unc/np.median(weighted_depths), color='k', fmt='o', markersize=10, markeredgecolor='k', ecolor=color'k', elinewidth=3, capsize=4, alpha=0.6)

    # Spitzer data
    spitzer_wave = 4.5      # microns
    spitzer_rprs = 0.0492
    spitzer_rprs_unc = 0.0008
    spitzer_depth = spitzer_rprs**2
    spitzer_depth_unc = spitzer_depth*np.sqrt(2.*((spitzer_rprs_unc/spitzer_rprs)**2))
    spitzer_label = 'Spitzer, Dittmann et al. (2017)'
    plt.errorbar(spitzer_wave, spitzer_depth*100., yerr=spitzer_depth_unc*100.,  fmt='o', color='firebrick', markersize=8, markeredgecolor='firebrick', ecolor='firebrick', elinewidth=2, capsize=0, alpha=0.8, label=spitzer_label)
    #plt.errorbar(spitzer_wave, spitzer_depth/np.median(weighted_depths), yerr = spitzer_depth_unc/np.median(weighted_depths), color='k', fmt='o', markersize=10, markeredgecolor='k', ecolor=color'k', elinewidth=3, capsize=4, alpha=0.6)

    plt.legend(loc='best')
    plt.xlim(.4, 5.)
    plt.xscale('log')
    plt.xticks(np.array([.4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5]), ('0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1', '2', '3', '4', '5'))
    #plt.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylabel('transit depth [%]')
    plt.xlabel('wavelength [microns]')
    plt.tight_layout()
    plt.show()
