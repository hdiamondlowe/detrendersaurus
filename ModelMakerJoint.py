import zachopy.Talker
Talker = zachopy.Talker.Talker
import numpy as np
from BatmanLC import BatmanLC

class ModelMakerJoint(Talker):

    def __init__(self, inputs, wavebin, params):
        ''' initialize the model maker'''
        Talker.__init__(self)

        self.inputs = inputs
        self.wavebin = wavebin
        self.params = params
    
    def makemodel(self):

        self.fitmodel = []
        for n, night in enumerate(self.inputs.nightname):
            x = []
            for flabel in self.inputs.fitlabels[n]:
                paramind = int(np.where(np.array(self.inputs.freeparamnames) == flabel+str(n))[0])
                x.append(self.params[paramind]*self.wavebin['compcube'][n][flabel])
            self.fitmodel.append(np.sum(x, 0) + 1)

        tranvalues = []
        for n, night in enumerate(self.inputs.nightname):
            values = {}
            for t, tranlabel in enumerate(self.inputs.tranlabels[n]):
                if tranlabel+str(n) in self.inputs.freeparamnames:
                    paramind = int(np.where(np.array(self.inputs.freeparamnames) == tranlabel+str(n))[0])
                    values[tranlabel] = self.params[paramind]
                elif self.inputs.tranbounds[n][0][t] == 'Joint':
                    jointset = int(self.inputs.tranbounds[n][1][t])
                    paramind = int(np.where(np.array(self.inputs.freeparamnames) == tranlabel+str(jointset))[0])
                    values[tranlabel] = self.params[paramind]
                else: values[tranlabel] = self.inputs.tranparams[n][t]   
            tranvalues.append(values)

        if self.inputs.istarget == True and self.inputs.isasymm == False:
            self.batmanmodel = []
            for n, night in enumerate(self.inputs.nightname):
                batman = BatmanLC(self.wavebin['compcube'][n]['bjd']-self.inputs.toff[n], tranvalues[n]['dt'], tranvalues[n]['rp'], tranvalues[n]['per'], tranvalues[n]['b'], tranvalues[n]['a'], tranvalues[n]['ecc'], tranvalues[n]['u0'], tranvalues[n]['u1'])
                batmanmodel = batman.batman_model()
                if np.all(batmanmodel == 1.): self.speak('batman model returned all 1s')
                self.batmanmodel.append(batmanmodel)
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
        return np.array(self.fitmodel)*np.array(self.batmanmodel)

