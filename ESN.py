from __future__ import division
import pdb
from matplotlib.pylab import *
import sys
from sklearn.linear_model import *
class ESN():

    """ General Echo State Network
	Parameters: 
		N_in: Input Dimension (currently 192) i.e steps per day 
                N_out: Output Dimension (currently 96) i.e steps per day minus nighttime
                N:     Network Size
                sparseness:  Sparseness of the reservoir
                transient_phase:  warm up time for the reservoir
                scaleW: scaling of reservoir to invoke fading memory
                leakrate: how leaky each unit is (set to 0 for 'normal' units)
                learnmode: method used for learning output weights options: {ridge,pinv}
    """
    def __init__(self,N_in = 192,N_out = 96, N=40, sparseness = 0.2,transient_phase=100,scaleW = 1.0,learnmode='pinv',leakrate=0):
        self.N_out = N_out
        self.N_in = N_in
        self.N = N
        # basic generation of weights of the ESN
        W  = np.random.rand(self.N,self.N)
        W[W > sparseness] = 0
        W[W != 0] = 2 * np.random.rand(W[W != 0].size) - 1
        self.W = W * scaleW
        self.W_in = 2 * np.random.rand(self.N,N_in) - 1
        self.W_out = 2 * np.random.rand(self.N+1,self.N_out) - 1 
        self.learnmode = learnmode 
        self.leakrate = leakrate
        # variables to increase runspeed
        self.initial_run = 1
        self.mins = 0
        self.maxs = 0
        self.rng = 0
        self.S = np.zeros((1,1))
        self.transient = transient_phase

    def forecast(self,Xold,Y,Xnow_old):
        if self.initial_run == 1:  # when the leaner is called the first time it runs with all historical data
            self.initial_run = 0   # so it is  transformed to reservoir states S 
            tmp = Xold
            self.mins = np.min(tmp, axis=0)   # in these steps the historical data is scaled such
            self.maxs = np.max(tmp, axis=0)   # they are between 0-1 and the scaling parameters are stored 
            self.rng = self.maxs - self.mins  # for future input
            tmp = 1 - (( (self.maxs - tmp)) / self.rng)
            X = tmp

            T = self.transient + X.shape[0]
            self.S = np.zeros((self.N,T))  
            self.S[:,0] =  np.random.rand(self.N)
            for i in range(1,T):
                
                if i < self.transient:                                    # warmup to invoke healthy reservoir dynamics
                    I = np.random.rand(self.N_in)			  # by applying noise
                else:
                        I = X[i-self.transient,:]
                   
                self.S[:,i] = self.leakrate*self.S[:,i-1] + np.tanh(np.dot(self.W,self.S[:,i-1]) + np.dot(self.W_in,I)) # update step for the reservoir
        
        Xnow = 1 - (( (self.maxs - Xnow_old)) / self.rng)                        # scale new data with parameters from in the initial run

        design = np.vstack((self.S[:,self.transient:],np.ones(self.S[:,self.transient:].shape[1]))).T           # design matrix for regression, add a line of ones

        ## learning (see learning modes)
        if self.learnmode == 'ridge':
            for j in range(0,self.N_out):                                         
                clf = Ridge(alpha=10)
                clf.fit(design, Y[:,j])
                self.W_out[:,j] = clf.coef_
  
        elif self.learnmode =='pinv':
            pinv_design = np.linalg.pinv(design)
            for j in range(0,self.N_out):                                             
                self.W_out[:,j] = np.dot(pinv_design,Y[:,j])

        net = self.leakrate*self.S[:,-1] + np.tanh(np.dot(self.W,self.S[:,-1]) + np.dot(self.W_in,Xnow)) # get reservoir response for current day
        self.S = np.hstack((self.S,net.reshape(self.N,-1)))  # store new network response
        output = np.zeros((self.N_out))

        for j in range(0,self.N_out):                                              # output of learning system: add a 1 to network state for constant input s0
            output[j] = dot(hstack((self.S[:,-1],np.ones(1))),self.W_out[:,j])
        
        output[output < 0] = 0  # 'hack' for avoiding negative predictons
        return output
        


