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
        self.paramvals = paramvals
    
    def makemodel(self):

        x = []
        for f in range(len(self.inputs.fitlabels)):
            x.append(self.paramvals[f]*self.wavebin['compcube'][self.inputs.fitlabels[f]])
        self.fitmodel = np.sum(x, 0) + 1

        tranvalues = {}
        for t, tranlabel in enumerate(self.inputs.tranlabels):
            if tranlabel in self.inputs.paramlabels:
                ind = np.where(np.array(self.inputs.paramlabels) == tranlabel)[0]
                if tranlabel == 'q0':
                    q0, q1 = paramvals[ind], paramvals[ind + 1]
                    tranvalues['u0'] = 2.*np.sqrt(q0)*q1
                    tranvalues['u1'] = np.sqrt(q0)*(1 - 2.*q1)
                elif tranlabel == 'q1': continue    
                else: tranvalues[tranlabel] = self.paramvals[int(ind)]
            else: 
                tranvalues[tranlabel] = self.inputs.tranparams[t]

        if self.inputs.istarget == True and self.inputs.isasymm == False:
            batman = BatmanLC(self.wavebin['compcube']['bjd']-self.inputs.toff, tranvalues['dt'], tranvalues['rp'], tranvalues['per'], tranvalues['b'], tranvalues['a'], tranvalues['ecc'], tranvalues['u0'], tranvalues['u1'])
            self.batmanmodel = batman.batman_model()
            if np.all(self.batmanmodel == 1.): self.speak('batman model returned all 1s')
        if self.inputs.istarget == True and self.inputs.isasymm == True:
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

