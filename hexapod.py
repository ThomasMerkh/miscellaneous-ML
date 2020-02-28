"""
tmerkh@ucla.edu, Thomas Merkh
This code was written as the controller script for the hexapod morphology (part of YARS, see Zahedi, Keyan, Arndt von Twickel, and Frank Pasemann. "Yars: A physical 3d simulator for evolving controllers for real robots." International Conference on Simulation, Modeling, and Programming for Autonomous Robots. Springer, Berlin, Heidelberg, 2008.)

This uses a conditional restricted boltzmann machine as the policy model, and learns a policy using the policy gradient method GPOMDP (Bartlett, Peter L., and Jonathan Baxter. "Infinite-horizon policy-gradient estimation." arXiv preprint arXiv:1106.0665 (2011).).

The learning algorithm is ADAM.

Note that in order to run this script, one needs to have YARS installed, see (https://sourceforge.net/projects/yars/, and http://yars.sourceforge.net/w/index.php/Main_Page).
"""

import math
import numpy as np
import numpy.matlib as npm
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from timeit import default_timer as timer

t = 0

def init():
    class MoD:
        def __init__(self):
            self.ex = 0          # 1 uses exact policy model, ~=1 uses Gibbs sampling
            self.rini = 1        # 0 intializes Gibbs chains with last action; 1 at random a; 2 uses a persistent Gibbs chain for Monte Carlo expectations 
            self.nrit = 10       # number of up-down Gibbs updates 
            self.nrst = 50       # number of samples for Monte Carlo expectations
            # input and output variables
            self.nb   = 2                    # number of bits per sensor/actuator variable
            self.nv   = 12                   # number of sensor variables
            self.vars = range(self.nv)       # selection of sensor variables 
            self.ni   = self.nv*self.nb      # number of input units
            self.no   = self.nv*self.nb      # number of output units  
            self.hid  = 5                    # number of hidden factors        
            self.nbh  = 5                    # number of bits per hidden factor
            # 'Hidden units' = hidden_bits*hidden_factors

            # connectivity structure
            # These hardcoded masks assume self.hid  = 6.
            self.Vmask = np.array([ [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]])
            self.Vmask = np.ones((self.hid,self.nv))  # Fully-Connected. If commented, CRBM connections are restricted as defined by the masks
            self.Vmask = np.kron(self.Vmask[:,self.vars],np.ones((self.nbh,self.nb))) 
            self.Wmask = np.array([ [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])

            self.Wmask = np.ones((self.hid,self.nv)) # Fully-Connected. If commented, CRBM connections are restricted as defined by the masks
            self.Wmask = np.kron(self.Wmask[:,self.vars],np.ones((self.nbh,self.nb))) 

            self.Vmask = self.Vmask[range(self.hid*self.nbh),:]  
            self.Wmask = self.Wmask[range(self.hid*self.nbh),:] 
            self.nh    = self.Vmask.shape[0]  # number of hidden units

    #  DEFINE TRAINING HYPERPARAMETERS
    class Configuration:
        def __init__(self):
            self.starttime = timer()
            self.randomstart = 0.10     # scale factor of randn parameter initialization
            self.T = 200                # number of iterations for average reward gradient estimation (GPOMDP)  
            self.be = 0.75              # discount factor in average estimation                                 
            self.learn = 0.001          # learning (stepsize) rate; 0.001 for Adam; 0.1 for simple momentum 
            self.xi = 1.00              # rewardf #1 exponent       
            self.optimizer = 2          # 1 == simple momentum/learning decay.  2 == ADAM (see parameter options in ADAM class)
            self.rewardf = 0            # 0: "r = rp", 1: "r = sign(rp)*rp^\xi rh^{1-\xi}"                        
            self.decay = 0              # optimizer == 1 param: decay of the learning rate over time; total rate is learn/(1+decay*t^.6)  
            self.momentum = 0.0         # optimizer == 1 param: weight of added previous gradients     
            self.weightdecay = 0.0001   # 0 means no weight decay 0.00001
            self.Tent = 130             # number of time steps for sliding window entropy estimation
            self.rh = 1                 # if ~0 then Config.rh times the average of the estimated entropy pro sensor is added to the reward
            self.gr = 1                 # IRF parameter telling how many variables to group.  1 == Factorized MI, self.ni == Joint state MI
            self.loadin     = 0         # 1 if the program should expect numpy files to load in for theta and shist - files in 'run' directory
            self.loadinnumber = 2000       # If loadin == 1, then this number indicates what timestep .npy files to load in.
            # save and visualize
            self.visualize  = 1         # if 1 make plots on the predefined schedule below
            self.Isomap     = True
            self.Iso_times  = [self.T*5+1]
            for i in range(68):
                self.Iso_times.append(self.T*100*(i+1)+1)
            self.iso_neighbors = 7      # Isomap Algorithm Parameter
            self.save_times = [self.T*2000 + 1, self.T*5000 + 1, self.T*8000 + 1]
            self.verbose    = 0         # if 1, print more output
            self.KILL       = self.save_times[-1] + 2        # Time step to kill simulation

    class Theta:
    	def __init__(self):
        	self.V = 0
        	self.c = 0
        	self.W = 0
        	self.b = 0

    class ADAM:
        def __init__(self):
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 0.00000001
            self.m_V = 0
            self.v_V = 0
            self.m_hat_V = 0
            self.v_hat_V = 0
            self.m_c = 0
            self.v_c = 0
            self.m_hat_c = 0
            self.v_hat_c = 0
            self.m_W = 0
            self.v_W = 0
            self.m_hat_W = 0
            self.v_hat_W = 0
            self.m_b = 0
            self.v_b = 0
            self.m_hat_b = 0
            self.v_hat_b = 0

        def configure(self, theta):
            self.m_V = np.zeros(theta.V.shape)
            self.v_V = np.zeros(theta.V.shape)
            self.m_hat_V = np.zeros(theta.V.shape)
            self.v_hat_V = np.zeros(theta.V.shape)

            self.m_c = np.zeros(theta.c.shape)
            self.v_c = np.zeros(theta.c.shape)
            self.m_hat_c = np.zeros(theta.c.shape)
            self.v_hat_c = np.zeros(theta.c.shape)

            self.m_W = np.zeros(theta.W.shape)
            self.v_W = np.zeros(theta.W.shape)
            self.m_hat_W = np.zeros(theta.W.shape)
            self.v_hat_W = np.zeros(theta.W.shape)

            self.m_b = np.zeros(theta.b.shape)
            self.v_b = np.zeros(theta.b.shape)
            self.m_hat_b = np.zeros(theta.b.shape)
            self.v_hat_b = np.zeros(theta.b.shape)

    Delta0 = Theta()
    Delta = Theta()
    Mod = MoD()
    Config = Configuration()
    # Ensure compatibility between parameters.
    if(Mod.ni % Config.gr != 0):
        print("Config.gr doesn't naturally divide Mod.ni! Mutual information can't be calculated. Exiting...")
        raise SystemExit
    theta = Theta()
    Adam = ADAM()
    theta.V = np.multiply((Config.randomstart * np.random.randn(Mod.nh,Mod.ni)),Mod.Vmask)   # hidden-input interactions
    theta.c = Config.randomstart*np.random.randn(Mod.nh,1)                                   # hidden bias
    theta.W = np.multiply((Config.randomstart * np.random.randn(Mod.nh,Mod.no)),Mod.Wmask)   # hidden-output interactions
    theta.b = Config.randomstart*np.random.randn(Mod.no,1)                                   # output bias

    # Parameters output.
    File = open('Parameters.txt','w')
    File.write(str(vars(Config)))
    File.write('\n\n\n')
    File.write(str(vars(Mod)))
    File.close()

    return Config, Mod, theta, Delta0, Delta, Adam


# YARS needs this defined in order to run
def close(): 	
  pass

# Classes that get instantiated at the beginning of every episode 
class E:
    def __init__(self):
        self.V = np.zeros(theta.V.shape)
        self.c = np.zeros(theta.c.shape)
        self.W = np.zeros(theta.W.shape)
        self.b = np.zeros(theta.b.shape)

def Isomapper():
    if(Config.verbose == 1):
        print("Performing 2D-Isomap with " + str(Config.iso_neighbors) + " neighbors.  This may take a minute...\n")

    data = np.asarray(Iso_Params)
    iso = manifold.Isomap(Config.iso_neighbors, n_components=2, eigen_solver='auto', path_method='auto', neighbors_algorithm='auto')
    isomapped_data = iso.fit_transform(data).T
    # Plot
    neigh = getattr(iso, "nbrs_")
    neighbors = neigh.kneighbors_graph() # Connectivity matrix.  Row indicates node, column indices with entry "1" indicate nearest neighbors
    neighbors = neighbors.toarray()
    G=nx.Graph()
    print(len(isomapped_data[0]))
    for i in range(len(isomapped_data[0])):
        G.add_node(i,pos=(isomapped_data[0][i],isomapped_data[1][i]))
    for i in range(len(isomapped_data[0])):
        neighborz = np.where(neighbors[i] == 1)[0]
        for j in range(neighborz.shape[0]):
            G.add_edge(i, neighborz[j])

    pos=nx.get_node_attributes(G,'pos')

    fig = plt.figure(figsize=(6, 6))
    plt.suptitle(" ", fontsize=14)
    ax = fig.add_subplot(111)
    nx.draw(G, pos, with_labels=False,node_size = 40, node_color = Iso_Rewards, cmap=plt.cm.rainbow) #Iso_Rewards for color to indicate reward ordering
    plt.title("Isomapped Parameter Space, colormap indicates reward")
    plt.axis('tight')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=np.min(Iso_Rewards), vmax=np.max(Iso_Rewards)))
    sm._A = []
    plt.colorbar(sm)
    plt.savefig("Isomap_Rewards.png")
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    plt.suptitle(" ", fontsize=14)
    ax = fig.add_subplot(111)
    nx.draw(G, pos, with_labels=False,node_size = 40, node_color = Config.iso_times, cmap=plt.cm.rainbow) #Config.iso_times for color to indicate time ordering
    plt.title("Isomapped Parameter Space, colormap indicates time step")
    plt.axis('tight')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=int(np.min(Config.iso_times)/Config.T), vmax=int(np.max(Config.iso_times)/Config.T)))
    sm._A = []
    plt.colorbar(sm)
    plt.savefig("Isomap_Times.png")
    plt.close()

    print("\nIsomapping Complete, continuing simulation\n")

def Plotting_Utility(t):
    # current CRBM parameters
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    axes[0, 0].imshow((theta.V), aspect='auto')
    axes[0, 1].imshow((theta.W), aspect='auto')
    axes[1, 0].imshow((theta.c), aspect='auto')
    axes[1, 1].imshow((theta.b), aspect='auto')
    axes[0, 0].set_title('theta.V')
    axes[0, 1].set_title('theta.W')
    axes[1, 0].set_title('theta.c')
    axes[1, 1].set_title('theta.b')
    plt.tight_layout()
    plt.savefig("theta" + str(t) + ".png")
    plt.close()

    # Recent Sensor history
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(np.linspace(0,Shist.shape[0],Shist.shape[0]),Shist[:,0], Shist[:,1])
    plt.title("Sensor_History")
    plt.savefig("Shist" + str(t) + ".png")
    plt.close()

    # Recent Action history
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(np.linspace(0,Ahist.shape[0],Ahist.shape[0]),Ahist[:,0], Ahist[:,1])
    plt.title("Action History")
    plt.savefig("Ahist" + str(t) + ".png")
    plt.close()

    ### Rewards over time
    sep = 10
    total_reward_points    = Rew[0:len(Rew):sep]
    external_reward_points = Rewp[0:len(Rew):sep]
    intrinsic_rwd_poitns   = RRH[0:len(Rew):sep]
    empirical_entropyy     = REE[0:len(Rew):sep]
    S1 = [total_reward_points[0]]
    S2 = [external_reward_points[0]]
    S3 = [intrinsic_rwd_poitns[0]]
    S4 = [empirical_entropyy[0]]
    ## Compute Exponentially moving average with \a = 2/(N+1), larger numbers is faster decay
    Alph1 = 12.0/(len(total_reward_points) + 1)
    Alph2 = 12.0/(len(external_reward_points) + 1)
    Alph3 = 12.0/(len(intrinsic_rwd_poitns) + 1)
    Alph4 = 12.0/(len(empirical_entropyy) + 1)
    for i in range(len(total_reward_points) - 1):
        S1.append( Alph1*total_reward_points[i+1] + (1.0-Alph1)*S1[i])
        S2.append( Alph2*external_reward_points[i+1] + (1.0-Alph2)*S2[i])
        S3.append( Alph3*intrinsic_rwd_poitns[i+1] + (1.0-Alph3)*S3[i])
        S4.append( Alph4*empirical_entropyy[i+1] + (1.0-Alph4)*S4[i])

    # Plot averages over measurements.
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(18,18))
    ax1.plot(np.linspace(0,sep*len(total_reward_points),len(total_reward_points)) , total_reward_points)
    ax1.plot(np.linspace(0,sep*len(total_reward_points),len(total_reward_points)) , S1)
    ax1.set_title('Total Reward')
    ax2.plot(np.linspace(0,sep*len(external_reward_points),len(external_reward_points)), external_reward_points)
    ax2.plot(np.linspace(0,sep*len(external_reward_points),len(external_reward_points)) , S2)
    ax2.set_title('Position Reward')
    ax3.plot(np.linspace(0,sep*len(intrinsic_rwd_poitns),len(intrinsic_rwd_poitns)),intrinsic_rwd_poitns)
    ax3.plot(np.linspace(0,sep*len(intrinsic_rwd_poitns),len(intrinsic_rwd_poitns)), S3)
    ax3.plot(np.linspace(0,sep*len(intrinsic_rwd_poitns),len(intrinsic_rwd_poitns)), S4, color='y')
    ax3.set_title('MI Reward with exponentially moving avgerage / Empirical Entropy of sensors in yellow')
    ax1.set_xlabel('Policy Update #')
    ax2.set_xlabel('Policy Update #')
    ax3.set_xlabel('Policy Update #')
    plt.savefig('Rewards' + str(t) + '.png')
    plt.close()

    # This part records the rewards in .csv files for later processing
    if(True):
        f=open("TotalReward.csv", "w")
        g=open("ExtReward.csv", "w")
        h=open("IntReward.csv", "w")
        for i in range(len(total_reward_points)):
            if(i < len(total_reward_points)-1):
                f.write(str(total_reward_points[i]) + ',')
                g.write(str(external_reward_points[i]) + ',')
                h.write(str(intrinsic_rwd_poitns[i]) + ',')
            else:
                f.write(str(total_reward_points[i]))
                g.write(str(external_reward_points[i]))
                h.write(str(intrinsic_rwd_poitns[i]))
        f.close()
        g.close()
        h.close()

    # On final save, print average reward
    if(t == Config.save_times[-1]):
        print("Average total reward at end of simulation:" , np.mean(S1[len(S1)-10 : len(S1)]))
        # Find when learning started!
        indx = 0
        found = False
        while (indx < len(S2) and not found):
            if(indx*sep < 2000):
                indx += 1
            elif(abs(S2[indx]) > 0.02):
                found = True
            else:
                indx += 1
        if(found):
            print("Locomotion was evident by policy update " + str(math.floor(indx*sep/Config.T)) )
        else:
            print("Locomotion never became evident.")

    ## Save Current Parameters and sensor/action histories for loading model
    np.save('thetaV_' + str(t), theta.V)
    np.save('thetaW_' + str(t), theta.W)
    np.save('thetac_' + str(t), theta.c)
    np.save('thetab_' + str(t), theta.b)
    np.save('Shist_' + str(t), Shist)
    np.save('Ahist_' + str(t), Ahist)
    return

def empentr(S,nb):
    # Compute Empirical Entropy of data streams contained in various columns
    H = np.zeros((S.shape[1],1))
    for i in range(S.shape[1]):
        # Empirical Distribution of i-th sensor
        suppt, counts = np.unique(S[:,i], return_counts=True)
        counts = 1.0*counts/S.shape[0]
        H[i] = -1.0*np.log(counts).dot(counts)
    # normalize H by maximum entropy and return
    H = H/np.log(pow(2,nb))
    return H[:,0]

def empMI(S,nb,gr):
    # Compute the empirical MI of the data streams contained in columns of S.
    # Values of S are assumed to be reals in [0,1]
    # The argument gr should be a natural the divides S.shape[1]
    #   It indicates how many variables to group; 1 computes the MI per column; S.shape computes the joint MI.
    # Verified as correct

    S = S.dot(np.kron(np.eye(int(S.shape[1]/gr)), np.power(2,nb*np.arange(gr))).T)
    MI = np.zeros((1,S.shape[1]))
    mil = np.zeros((1,S.shape[1]))
    for i in range(S.shape[1]):
        # Determine the empirical joint distribution of the i-th sensor at times t and t-1.
        C, ia, ic = np.unique( np.asarray([ S[0:S.shape[0]-1, i], S[1:S.shape[0]+1, i ] ]).T , axis=0, return_inverse=True, return_index=True)
        pjt = np.zeros((C.shape[0] , 1))
        for j in range(C.shape[0]):
            pjt[j] = len(np.where(ic == j)[0])
        pjt = pjt/(S.shape[0]-1)
        pjt = pjt[ic]
        
        # Determine the empirical distribution of the i-th sensor at time t.
        C1, ia1, ic1 = np.unique(S[0:S.shape[0]-1, i], return_inverse=True, return_index=True)
        p1t = np.zeros((C1.shape[0] , 1), dtype=float)
        for j in range(C1.shape[0]):
            p1t[j] = len(np.where(S[0:S.shape[0]-1, i] == C1[j])[0] )
        p1t = p1t/(S.shape[0]-1)
        p1t = p1t[ic1]

        # Determine the empirical distribution of the i-th sensor at time t+1
        C2, ia2, ic2 = np.unique( S[1:S.shape[0]+1,i], return_inverse=True, return_index=True)
        p2t = np.zeros((C2.shape[0] , 1), dtype=float)
        for j in range(C2.shape[0]):
            p2t[j] = len(np.where(S[1:S.shape[0]+1,i] == C2[j])[0] )
        p2t = p2t/(S.shape[0]-1)
        p2t = p2t[ic2]

        MI[0,i] = np.mean(np.log( np.divide(pjt,np.multiply(p1t, p2t))))
        indi = np.where(ic1 == ic1[len(ic1)-1])[0]

        mil[0,i] = np.mean(np.log( np.divide( pjt[indi] , np.multiply(p1t[indi],p2t[indi]))))

    MI = MI/np.log(pow(2, nb*gr))
    mil = mil/np.log(pow(2, nb*gr))
    MI = np.kron(MI,np.ones((1,gr)))
    mil = np.kron(mil,np.ones((1,gr)))

    return MI, mil


def binarize(S,nb):
    nv = S.shape[0]
    S = S*pow(2,nb)
    S = np.abs(np.floor(S)).astype(int)
    binarynumbers = []
    maxlength = 0
    # Now convert each number into binary
    for i in range(nv):
        binarynumbers.append("{0:b}".format(S[i]))
        if(len(binarynumbers[i]) > maxlength):
            maxlength = len(binarynumbers[i])
    if(maxlength < nb):
        maxlength = nb
    for i in range(nv):
        while(len(binarynumbers[i]) < maxlength):
            binarynumbers[i] = '0' + binarynumbers[i]
    # We are looking for a column vector of length nb*nv with entries 1 or 0
    s = np.zeros((nb*nv,1))
    for i in range(nb*nv):
        s[i,0] = (int(binarynumbers[int(i/maxlength)][i % maxlength]))
    return s

def dec2bin(i, length):
    # This is a python modified version of the MATLAB function dec2bin
    # Ensures that the binary represenation of 'i' is of length 'length'. Returns a string
    if (i == 0): 
        return "0"*length
    s = ''
    while(i > 0):                 # Continues until i = 0.
        if int(i) & 1 == 1:       # only true if i == 1.
            s = "1" + s
        else:
            s = "0" + s
        i = int(i/2)
    while(len(s) < length):
        s = "0" + s
    return s

def binlist(V):
    # Returns a 2D array with rows being every binary string of length V
    v = np.zeros((pow(2,V),V))
    for i in range(pow(2,V)):
        strg = dec2bin(i,V)
        for j in range(V):
            v[i,j] = int(strg[j])
    return v

def draw(p):
    # Draws a sample from a probability np.array p
    K = np.random.random() <= np.cumsum(p)
    T = np.where( K == 1 )
    return T[0].min()  # returns the first index.

def realize(a,nb):
    # Produces a vector with a real entry on [0,1] per nb bits of binary np.array 'a'
    # e.g. if a.shape = (24,1), and nb = 2, then len(A) = 12

    nv = int(a.shape[0]/nb)
    C = np.asarray(list(map(lambda x: 2**float(x), np.arange(-1,-nb-1,-1)))).reshape(1,nb)
    B = np.reshape(a, (nb,nv), order='F')
    return C.dot(B) + 0.5/(pow(2,nb))

def sigmp(x):
    # Takes in an np.array, outputs the element wise sigmoid function as np.array
    return 1.0/(1 + np.exp(-1.0*x)) 

def drawCRBM(a,s):
    # Draw an action 'a' from CRBM given the sensor value s, and the parameter theta.
    if(Mod.ex == 1):
        # Do exact computations when there are few variables
        H = binlist(Mod.nh)
        A = binlist(Mod.no)
        exponent = np.exp(npm.repmat((H.dot(theta.V.dot(s))), 1, A.shape[0]) + npm.repmat(H.dot(theta.c), 1, A.shape[0]) + H.dot(theta.W.dot(np.transpose(A))) + npm.repmat(np.transpose(theta.b).dot(np.transpose(A)), H.shape[0],1))
        # Marginalize by summing over first dimension
        p = exponent.sum(axis = 0)
        p = p / np.linalg.norm(p,1)
        a0 = np.zeros((Mod.no,1))
        temp = dec2bin(int(draw(p)),Mod.no)
        for i in range(Mod.no):
            a0[i] = int(temp[i])
    else:
        # Do Gibbs sampling if CRBM is too big for direct computation
        if(Mod.rini == 1 or Mod.rini == 2):
            a0 = np.floor(np.random.rand(Mod.no,1) + 0.5*np.ones((Mod.no,1)))
        else:
            a0 = np.zeros((len(a),1))
            a0[:,0] = a[:,0]

        for it in range(Mod.nrit):
            h = (sigmp(theta.V.dot(s) + theta.c + theta.W.dot(a0)) > np.random.rand(Mod.nh,1))*1.0
            a0 = (sigmp(np.transpose(theta.W).dot(h) + theta.b) > np.random.rand(Mod.no,1))*1.0                    
    return a0


def gradCRBM(a,s):
    # Calculates the gradient of the policy model - adjusts values of dtheta
    if(Mod.ex == 1): #Compute the exact gradient, no Gibbs sampling.
        A = binlist(Mod.no)
        H = binlist(Mod.nh)
        # p(h | s, a)
        psa = np.exp(H.dot(theta.V.dot(s)) + H.dot(theta.c) + H.dot(theta.W.dot(a)))
        psa = psa/psa.sum(axis=0)
        # p(h, a | s)
        ps = np.exp(npm.repmat(H.dot(theta.V.dot(s)), 1, A.shape[0]) + npm.repmat(H.dot(theta.c), 1, A.shape[0]) + H.dot(theta.W.dot(np.transpose(A))) + npm.repmat(np.transpose(theta.b).dot(np.transpose(A)),H.shape[0],1))
        ps = ps/ps.sum()

        # Numpy has does 1D and 2D (n by 1) mathematics differently.  Be careful (thats why [:,0] appears below)
        dtheta.V = np.outer(np.transpose(H).dot(psa), np.transpose(s)) - np.outer(np.transpose(H).dot(psa.sum(axis=1)) , np.transpose(s))
        dtheta.c[:,0] = np.transpose(H).dot(psa)[:,0] - np.transpose(H).dot(ps.sum(axis=1))
        dtheta.W = np.outer(np.transpose(H).dot(psa), np.transpose(a)) - np.transpose(H).dot(ps.dot(A))
        dtheta.b[:,0] = a[:,0] - np.transpose( ps.sum(axis=0).dot(A) )

    else:
        #If many variables, do Gibbs Sampling and MC averaging to approximate gradient
        nrst = Mod.nrst  # Number of Samples for Monte Carlo Means
        nrit = Mod.nrit  # Number of updown passes

        ## Pr(H = 1|a,s), repeated nrst times (which will be averaged) # Previously was using nrst samples of (h | s,a), instead of probabiliies.
        ## See contrastive divergence algorithm as to why probabilities should be used.
        h2 = npm.repmat(sigmp(theta.V.dot(s) + theta.c + theta.W.dot(a)),1,nrst)*1.0

        # Gather nrst samples of (h,a) given s via Gibbs Sampling
        if(Mod.rini == 2):
            # Initialize first Gibbs chain at random, and use its output to initialize the next, etc.
            a0 = (np.random.rand(Mod.no,1) > 0.5)*1.0
            h1 = np.zeros((Mod.nh,nrst))
            a1 = np.zeros((Mod.no,nrst))
            for st in range(nrst):   
                for it in range(nrit):  
                    h  = (np.random.rand(Mod.nh,1) < sigmp(theta.V.dot(s) + theta.c + theta.W.dot(a0)))*1
                    a0 = (np.random.rand(Mod.no,1) < sigmp(np.transpose(theta.W).dot(h) + theta.b))*1
                ## For last update use probabilities
                h0 = sigmp(theta.V.dot(s) + theta.c + theta.W.dot(a0))
                a0 = sigmp(np.transpose(theta.W).dot(h) + theta.b)
                # Collect Samples
                h1[:,st] = h0[:,0]
                a1[:,st] = a0[:,0]
            h0 = h1
            a0 = a1
        else:
            # Run all Gibbs chains in parallel
            if(Mod.rini == 1):   # Initialize each chain at random
                a0 = (np.random.rand(Mod.no,nrst) > 0.5)*1.0
            elif(Mod.rini == 0): # Initialize each chain at current action
                a0 = a.dot(np.ones((1,nrst)))

            for it in range(nrit):  
                h  = (np.random.rand(Mod.nh,nrst) < sigmp(theta.V.dot(s.dot(np.ones((1,nrst)))) + theta.c.dot(np.ones((1,nrst))) + theta.W.dot(a0)))*1.0
                a0 = (np.random.rand(Mod.no,nrst) < sigmp(np.transpose(theta.W).dot(h) + theta.b.dot(np.ones((1,nrst)))))*1.0
            ## For last update use probabilities
            h0 = sigmp(theta.V.dot(s.dot(np.ones((1,nrst)))) + theta.c.dot(np.ones((1,nrst))) + theta.W.dot(a0))
            a0 = sigmp(np.transpose(theta.W).dot(h) + theta.b.dot(np.ones((1,nrst))))
       
        # Update dtheta
        dtheta.V = np.multiply( ( (np.outer(h2.sum(axis=1),np.transpose(s)) - np.outer(h0.sum(axis=1),np.transpose(s)))/float(nrst)), Mod.Vmask)
        dtheta.c[:,0] = (h2.sum(axis=1) - h0.sum(axis=1))/float(nrst)
        dtheta.b[:,0] = (a[:,0] - a0.sum(axis=1))/float(nrst)
        dtheta.W = np.multiply( ((np.outer(h2.sum(axis=1),np.transpose(a)) - (h0.dot(np.transpose(a0))))/float(nrst)), Mod.Wmask)
    return

def update(sensors):
    global t
    t = t + 1   

    if(t == 1):
        # Allows classes to exist outside of the first time step
        global Config, Mod, theta, Delta0, Delta, Adam
        global xpos0, ypos0, Rew, Rewp, RRH, a, Shist, Ahist, A, REE, Iso_Params, Iso_Rewards

        # Initialize classes that persist throughout the entire program
        Config, Mod, theta, Delta0, Delta, Adam = init()

        if(Config.optimizer == 2):
            Adam.configure(theta)

        xpos0 = sensors[12]
        ypos0 = sensors[13]
        Rew = []    
        Rewp = []   
        RRH = [] 
        REE = []   
        Iso_Params = []
        Iso_Rewards = []
        a = np.zeros((Mod.no,1))
        A = [0]*Mod.nv
        Shist = np.zeros((Config.Tent,Mod.nv))
        Ahist = np.zeros((Config.Tent,Mod.nv))

        if(Config.loadin == 1):     
            theta.V = np.load('thetaV_' + str(Config.loadinnumber) + '.npy')
            theta.c = np.load('thetac_' + str(Config.loadinnumber) + '.npy')
            theta.W = np.load('thetaW_' + str(Config.loadinnumber) + '.npy')
            theta.b = np.load('thetab_' + str(Config.loadinnumber) + '.npy')
            Shist   = np.load('Shist_' +  str(Config.loadinnumber) + '.npy')
            print("Load-in Success")

        if(Config.verbose == 1):
            print("\nInput/Hidden/Output:", str(Mod.ni) + "/" + str(Mod.nh) + "/" + str(Mod.no))
            print("Time steps between policy updates (Config.T):", Config.T)
            print("Exact Gradient (1 == True, 0 == False):", Mod.ex)
            print("Optimizer information:")
            if(Config.optimizer == 1):
                print("Using simple momentum with weight decay")
                print("Learning Rate:", Config.learn)
                print("Momentum:", Config.momentum)
                print("Weight Decay:", Config.weightdecay)
            elif(Config.optimizer == 2):
                print("Using Adam with parameters:")
                print("Learning rate =", Config.learn)
                print("First Moment Decay Beta_1 = ", Adam.beta1)
                print("Second Moment Decay Beta_2 = ", Adam.beta2)
                print("Epsilon = ", Adam.epsilon)
            if(Mod.rini == 0 and Mod.ex == 0):
                print("Mod.rini == 0, using previous action to initialize Gibbs Chains")
            elif(Mod.rini == 1 and Mod.ex == 0):
                print("Mod.rini == 1, using random action to initialize Gibbs Chains")
            elif(Mod.rini == 2 and Mod.ex == 0):
                print("Mod.rini == 2, using persistant Gibbs Chains")
            if(Mod.ex == 0):
                print("Using", Mod.nrit, "up-down sweeps for Gibbs chains")
                print("Using", Mod.nrst, "samples for Monte-Carlo Averages")
            print("Simulation will run for", Config.save_times[-1], "time steps")
            if(Config.Isomap == True):
                print("Isomap will be done at simulation's end.")

        if(Config.Isomap == True):
            # Check that there will be more isomapping datapoints than nearest neighbors
            if(len(Config.Iso_times) < Config.iso_neighbors):
                print("Not enough Config.Iso_times selected, must be greater than number of nearest neighbors.")
                print("Please fix before continuing... Exiting Simulation")
                raise SystemExit
        
        print("\nDone Initialization")

    else:
        tt = t-1
        if(Config.loadin == 1):
            tt = tt + (Config.loadinnumber+1)*Config.T
        if(tt % Config.T == 1):
            # Initialize variables that persist through a single episode only
            global e, hdR, dtheta0, time, rr, rrp, rrh, dtheta, ree
            e       = E()        # Eligibility Trace
            hdR     = E()        # The parameter update
            dtheta0 = E()        # Initialize gradients for previous sensor and action 
            dtheta  = E()        # Temporary variable
            time = 0             # initialize time for gradient estimation
            
            # initialize average reward for average reward estimation
            rr = 0; rrp = 0; rrh = 0; ree = 0;

        ##### Gradient Estimation #####
        time = time + 1
        # Reward for forward movement after previous state and action
        xpos = sensors[12] # x-coordinate of center of gravity (x = -sin(theta))
        ypos = sensors[13] # y-coordinate of center of gravity (y = -cos(theta))
        zori = sensors[17] # angle theta about the z-axis. 
        rp = (xpos-xpos0)*math.sin(math.pi*zori/180.0) - (ypos-ypos0)*math.cos(math.pi*zori/180.0)
        xpos0 = xpos
        ypos0 = ypos

        # Add current state to sensor history (to the bottom of 2D array Shist)
        Shist = np.delete(Shist,0,0)
        Shist = np.insert(Shist,[Shist.shape[0]], np.floor(np.array(sensors[0:Mod.nv])*pow(2,Mod.nb)), 0)       

        # Estimate MI Reward, this should be done once there's a sufficient sensor history
        if(Config.rh != 0 and t >= Config.Tent):
            rh = Config.rh * np.mean(empMI(Shist,Mod.nb,Config.gr)[0]) 
            re = Config.rh * np.mean(empentr(Shist,Mod.nb))
        else:
            rh = 0
            re = 0

        # Total reward at this time step.
        if(Config.rewardf == 0):
            r = rp
        elif(Config.rewardf == 1):
            # This reward function has a significantly smaller reward signal than just extrinsic, so a scaling factor can be used.
            # The scale factor seen here has been specifically determined for the hexapod locomotion task.
            r = 50*np.sign(rp)*pow( pow(abs(rp),Config.xi)*pow(rh,1-Config.xi) , 2)

        # Update Aux Variable using gradient at previous state and action
        e.V = Config.be * e.V + dtheta0.V
        e.c = Config.be * e.c + dtheta0.c
        e.W = Config.be * e.W + dtheta0.W
        e.b = Config.be * e.b + dtheta0.b
        # Update gradient estimate for previous state and action and obtained reward
        hdR.V = hdR.V + (r*e.V - hdR.V)/(time + 1.0)
        hdR.c = hdR.c + (r*e.c - hdR.c)/(time + 1.0)
        hdR.W = hdR.W + (r*e.W - hdR.W)/(time + 1.0)
        hdR.b = hdR.b + (r*e.b - hdR.b)/(time + 1.0)

        ##### End of gradient estimation #####

        #####      Update Variables      #####
        rr  = rr  + r    # Total Reward
        rrp = rrp + rp   # Position Reward
        rrh = rrh + rh   # Mutual information reward
        ree = ree + re   # Empirical Entropy reward

        # Get and binarize current sensor states
        s = binarize(np.array(sensors[0:Mod.nv]),Mod.nb)

        # draw action for the current sensor state (use last action a to initialize Gibbs sampling)
        a = drawCRBM(a,s)

        # Do a random action every 50 time steps.  Acts as regularizer and ensured exploration
        if(t % 50 == 3):
            a = np.random.choice([0,1], (Mod.no,1))

        # gradient of the policy model at current sensor state s and action a, divided by p(a|s)
        gradCRBM(a,s)
        dtheta0 = dtheta

        # Gradient Step
        if(time == Config.T):
            # Prepare parameter update
            if(Config.optimizer == 1): # Regular momentum + Learning rate decay.
                Delta.V = Config.learn*(hdR.V - Config.weightdecay * theta.V )/(1.0 + Config.decay*pow(tt,0.6)) + Config.momentum * Delta0.V
                Delta.c = Config.learn*(hdR.c - Config.weightdecay * theta.c )/(1.0 + Config.decay*pow(tt,0.6)) + Config.momentum * Delta0.c
                Delta.W = Config.learn*(hdR.W - Config.weightdecay * theta.W )/(1.0 + Config.decay*pow(tt,0.6)) + Config.momentum * Delta0.W
                Delta.b = Config.learn*(hdR.b - Config.weightdecay * theta.b )/(1.0 + Config.decay*pow(tt,0.6)) + Config.momentum * Delta0.b
            elif(Config.optimizer == 2): # Adam
                Adam.m_V     = Adam.beta1*Adam.m_V + (1-Adam.beta1)*hdR.V
                Adam.v_V     = Adam.beta2*Adam.v_V + (1-Adam.beta2)*(hdR.V**2)
                Adam.m_hat_V = Adam.m_V/(1-Adam.beta1**int(t/Config.T))
                Adam.v_hat_V = Adam.v_V/(1-Adam.beta2**int(t/Config.T))
                Adam.m_c     = Adam.beta1*Adam.m_c + (1-Adam.beta1)*hdR.c
                Adam.v_c     = Adam.beta2*Adam.v_c + (1-Adam.beta2)*(hdR.c**2)
                Adam.m_hat_c = Adam.m_c/(1-Adam.beta1**int(t/Config.T))
                Adam.v_hat_c = Adam.v_c/(1-Adam.beta2**int(t/Config.T))
                Adam.m_W     = Adam.beta1*Adam.m_W + (1-Adam.beta1)*hdR.W
                Adam.v_W     = Adam.beta2*Adam.v_W + (1-Adam.beta2)*(hdR.W**2)
                Adam.m_hat_W = Adam.m_W/(1-Adam.beta1**int(t/Config.T))
                Adam.v_hat_W = Adam.v_W/(1-Adam.beta2**int(t/Config.T))
                Adam.m_b     = Adam.beta1*Adam.m_b + (1-Adam.beta1)*hdR.b
                Adam.v_b     = Adam.beta2*Adam.v_b + (1-Adam.beta2)*(hdR.b**2)
                Adam.m_hat_b = Adam.m_b/(1-Adam.beta1**int(t/Config.T))
                Adam.v_hat_b = Adam.v_b/(1-Adam.beta2**int(t/Config.T))

                Delta.V = Config.learn*Adam.m_hat_V/(np.sqrt(Adam.v_hat_V) + Adam.epsilon)
                Delta.c = Config.learn*Adam.m_hat_c/(np.sqrt(Adam.v_hat_c) + Adam.epsilon)
                Delta.W = Config.learn*Adam.m_hat_W/(np.sqrt(Adam.v_hat_W) + Adam.epsilon)
                Delta.b = Config.learn*Adam.m_hat_b/(np.sqrt(Adam.v_hat_b) + Adam.epsilon)
            else:
                print("Choose a correct parameter for the optimizer... Exiting")
                raise SystemExit

            # parameter update
            theta.V = theta.V + Delta.V
            theta.c = theta.c + Delta.c
            theta.W = theta.W + Delta.W
            theta.b = theta.b + Delta.b
            # Update previous parameter update for momentum
            Delta0 = Delta

            # record estimated average reward
            Rew.append(rr/Config.T) 
            Rewp.append(rrp/Config.T)
            RRH.append(rrh/Config.T)
            REE.append(ree/Config.T)

            # Kills program at specified time
            # Placed here to increase efficiency
            if(t > Config.KILL):
                end = timer()
                print("Entire Program ran for:", end - Config.starttime, "seconds.\n")
                raise SystemExit
            elif(Config.verbose == 1):
                print("\n_______________________________")
                print("\nGradient Step:", int(tt/Config.T))  
                print("\n2-Norm of updates for (V,c,W,b):", '%.4f'%(np.linalg.norm(Delta.V,2)),'%.4f'%(np.linalg.norm(Delta.c,2)),'%.4f'%(np.linalg.norm(Delta.W,2)),'%.4f'%(np.linalg.norm(Delta.b,2)))
                print("\n2-Norm of theta.(V,c,W,b):", '%.4f'%(np.linalg.norm(theta.V,2)),'%.4f'%(np.linalg.norm(theta.c,2)),'%.4f'%(np.linalg.norm(theta.W,2)),'%.4f'%(np.linalg.norm(theta.b,2)))
                print("\nAverage rewards over last Config.T time steps (total, extrinsic, intrinsic):", '%.4f'%(1.0*rr/time), '%.4f'%(1.0*rrp/time), '%.4f'%(1.0*rrh/time) )
                print("\nPrevious Real-Valued Action:", [ '%.2f' % elem for elem in A[0:Mod.nv] ])
                print("\nCurrent Real-Valued Sensory Input:", [ '%.2f' % elem for elem in sensors[0:Mod.nv] ])
                print("\nCurrent Binarized Sensory Input:", s.T)
                print("\nCurrent CRBM actuator output:", a.T)
            else:
                print("Gradient Step:", int(tt/Config.T))  

        # Prepare actuator values for YARS
        A = realize(a,Mod.nb)
        A = A.tolist()[0]

        # Shift all the rows up in actuator history, and put state A as last row
        for Q in range(1,Config.Tent):
            Ahist[Q-1,:] = Ahist[Q,:]
        Ahist[Config.Tent-1,:] = A

        # Visualize and Save
        if(Config.visualize == 1 and tt in Config.save_times):
            print("Saving Data and plotting...")
            Plotting_Utility(int(tt/Config.T))
            print("Saving and plotting complete, continuing simulation \n")
        ## Store the current CRBM parameters and reward if its one of the prespecified time steps
        if(tt in Config.Iso_times and Config.Isomap == True and len(Rew) > 0):
            curr_param = []
            for i in range(theta.V.shape[0]):
                for j in range(theta.V.shape[1]):
                    curr_param.append(theta.V[i,j])
            for i in range(theta.b.shape[0]):
                for j in range(theta.b.shape[1]):
                    curr_param.append(theta.b[i,j])
            for i in range(theta.W.shape[0]):
                for j in range(theta.W.shape[1]):
                    curr_param.append(theta.W[i,j])
            for i in range(theta.c.shape[0]):
                for j in range(theta.c.shape[1]):
                    curr_param.append(theta.c[i,j])
            Iso_Params.append(curr_param)   # This adds the current parameters to be isomapped.
            if(t == Config.Iso_times[-1]):  # At the last time in Config.iso_times, do the actual mapping
                Isomapper()

    return A

# Old - new intrinsic reward implementation.


#def empMI(S,nb):
#    # Compute Empirical distribution p_t(s') of data streams contained in the columns.
#    Empirical_dist = np.zeros((S.shape[1],pow(2,nb)))       # number of sensors by the number of states.
#
#    # For each sensor...
#    for i in range(Empirical_dist.shape[0]): 
#        # For each possible sensor state... Ignore the current (last) sensory input
#        t = list(S[0:S.shape[0]-1,i])
#        for j in range(Empirical_dist.shape[1]):
#            # Count how many times state j was seen over last Config.Tent-1 samples.
#            Empirical_dist[i,j] = t.count(j)
#    # Now the number of times each sensor saw each state is recorded in empirical_dist.
#    # To get the percentage of time each sensor spent in each state, divide these counts by Config.Tent
#    Empirical_dist = Empirical_dist/(S.shape[0]-1)
#
#    # Compute the Empirical Conditional Distribution of data streams contained in the columns.
#    # Here its only conditioning on a single previous value, although extending this wouldn't be difficult.
#    possible_states = range(pow(2,nb))                           # [0,1]
#    last_reading = S[S.shape[0]-1,:]                             # The sensor state to condition on.
#    Empirical_Cond_dist = np.zeros((S.shape[1],len(possible_states)))    # This will hold the probabilities of seeing each sensor in state s' next.
#    
#    # For each sensor...
#    for i in range(Empirical_Cond_dist.shape[0]):
#        t = S[:,i]                           # Isolate just this sensor's past observations
#        # For each possible sensor state...
#        for j in possible_states:
#            # This line first creates a list of tuples containing consecutive observations.  Then it counts how often 
#            # state j is followed by the state we are conditioning on. 
#            Empirical_Cond_dist[i,j] = list(zip(t,t[1:])).count( (j, last_reading[i]))
#        # The rows need to be normalized to be probabilities
#        Empirical_Cond_dist[i,:] /= Empirical_Cond_dist[i,:].sum()
#   
#    # Computes the condition entropy for each sensor seperately
#    CondEnt = np.zeros(Empirical_dist.shape[0])
#    # States are columns, so we just have to dot product each row together.
#    for i in range(Empirical_dist.shape[0]):
#        for j in range(Empirical_dist.shape[1]):
#            # Make sure we aren't getting Nan's
#            if(Empirical_dist[i,j] > 10e-8 and Empirical_Cond_dist[i,j] > 10e-8):
#                CondEnt[i] += np.log(Empirical_Cond_dist[i,j]/Empirical_dist[i,j])*Empirical_Cond_dist[i,j]
#
#    return CondEnt/np.log(pow(2,nb)) # Scale by maximum entropy
