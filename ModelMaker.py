import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np
from BatmanLC import BatmanLC

class ModelMaker(Talker):

    def __init__(self, inputs, wavebin, paramvals):
        ''' initialize the model maker'''
        Talker.__init__(self)

        self.inputs = inputs
        self.wavebin = wavebin
        self.params = paramvals
    
    def makemodel(self):

        poly = []
        for plabel in self.inputs.polylabels:
            paramind = int(np.where(np.array(self.inputs.freeparamnames) == plabel)[0])
            poly.append(self.params[paramind])
        N = len(poly)
        polymodel = 0
        while N > 0:
            polymodel = polymodel + poly[N-1]*(self.wavebin['compcube']['bjd']-self.inputs.toff)**(N-1)
            N = N -1
        x = []
        for flabel in self.inputs.fitlabels:
            paramind = int(np.where(np.array(self.inputs.freeparamnames) == flabel)[0])
            x.append(self.params[paramind]*self.wavebin['compcube'][flabel])
        parammodel = np.sum(x, 0)
        self.fitmodel = polymodel + parammodel + 1

        tranvalues = {}
        for t, tranlabel in enumerate(self.inputs.tranlabels):
            if tranlabel in self.inputs.freeparamnames:
                paramind = int(np.where(np.array(self.inputs.freeparamnames) == tranlabel)[0])
                tranvalues[tranlabel] = self.params[paramind]
            else: tranvalues[tranlabel] = self.inputs.tranparams[t]   

        #print self.params

        if self.inputs.istarget == True and self.inputs.isasymm == False:
            batman = BatmanLC(self.wavebin['compcube']['bjd'], self.inputs.toff+tranvalues['dt'], tranvalues['rp'], tranvalues['per'], tranvalues['inc'], tranvalues['a'], tranvalues['ecc'], tranvalues['u0'], tranvalues['u1'])
            self.batmanmodel = batman.batman_model()
            if np.all(self.batmanmodel == 1.): self.speak('batman model returned all 1s')
        elif self.inputs.istarget == True and self.inputs.isasymm == True:
            rp, tau0, tau1, tau2 = [], [], [], []
            numtau = 0
            for k in tranvalues.keys():
                if 'tau' in k: numtau += 1
            numdips = numtau/3
            alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
            for i in range(numdips):
                rp.append(tranvalues['rp'+alphabet[i]])
                tau0.append(tranvalues['tau0'+alphabet[i]])
                tau1.append(tranvalues['tau1'+alphabet[i]])
                tau2.append(tranvalues['tau2'+alphabet[i]])
            t, F = self.wavebin['compcube']['bjd']-self.toff-tranvalues['dt'], tranvalues['F']
            for i in range(len(tau0)):
                F -= 2.*rp[i] * (np.exp((t-tau0[i])/tau2[i]) + np.exp(-(t-tau0[i])/tau1[i]))**(-1)
            self.batmanmodel = F
        elif self.inputs.istarget == False: 
            self.batmanmodel = np.ones(len(self.wavebin['compcube']['bjd']))
        #if self.dividewhite == True: return self.fit_model*self.batman_model*self.Zwhite[self.binnedok]*self.Zlambdat[self.binnedok]
        return self.fitmodel*self.batmanmodel

