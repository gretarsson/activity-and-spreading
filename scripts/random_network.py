import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import networkx as nx
from math import pi
from numba import jit
from feedback_helpers import feedback, compute_phase_coherence, plot_feedback
from tqdm import tqdm
import pickle
colours = sns.color_palette('muted')
matplotlib.rcParams.update({'font.size': 15, 'axes.labelsize':24})

# ---------------------------------
# Here we simulate a one-species
# heterodimer model
# -----------------------------------
fig_save_path = '../plots/'
np.random.seed(0)
run = True
# heterodimer parameters
N=10;p=0.50 # 0.75
rho=0.1;a0=1.0;ai=1;aii=1;api=1.1
eps=0.001;delta=10*eps;K=0.1/N;c=0.5

# drifting natural frequency setting
#w = np.random.normal(loc=10.0,scale=2.5,size=N)
#w[0] = 5.0

# phase-locked natural frequency setting
w = np.random.normal(loc=10.0,scale=0,size=N)

# changing parameter
M = 50
apis = np.linspace(0.75,1.5,M)
apis = np.append(apis, [1.0])
M = 51

# solver settings
t_span=(0,200);method='RK45';atol=1e-6;rtol=1e-3;max_step=1e-2

# initial conditions
y0_u = np.random.normal(loc=1.0,scale=0.0,size=N)
y0_up = np.random.normal(loc=1e-3,scale=0.00,size=N)
y0_theta = np.random.normal(loc=0.5,scale=0.25,size=N)
y0 = np.concatenate([y0_u, y0_up, y0_theta])

# create graph and related matrices
G = nx.erdos_renyi_graph(N,p,directed=False,seed=0)
L = nx.laplacian_matrix(G).toarray().astype(np.float64)
A = nx.adjacency_matrix(G).toarray().astype(np.float64)
pos = nx.circular_layout(G)
nx.draw(G,pos,node_color=colours[0:N])
plt.savefig(fig_save_path+'graph.pdf',dpi=300)

# find bifurcation point in api
us = np.empty((N,M))
ups = np.empty((N,M))
coherences = np.empty((M,))
if run:
    for m,api in enumerate(tqdm(apis)):
        # solve system
        sol = solve_ivp(lambda t, y: feedback(t, y, L, A, rho, a0, ai, aii, api, eps, delta, K, c, w), \
                                 t_span, y0, method=method, atol=atol, rtol=rtol, max_step=max_step)
        
        # extract and store steady-state
        u = sol.y[0:N,-1]
        up = sol.y[N:2*N,-1]
        theta = sol.y[2*N:3*N,:]
        us[:,m] = u
        ups[:,m] = up
        coherences[m] = np.mean(compute_phase_coherence(theta))

    # save store
    with open('../simulations/random_network.pkl', 'wb') as f:
        pickle.dump((us, ups, coherences), f)

# load store
with open('../simulations/random_network.pkl', 'rb') as f:
    us, ups, coherences = pickle.load(f)
    
# plot steady=state
fontsize=20
plt.style.use('seaborn-muted')
plt.figure()
for k in range(N):
    plt.plot(apis,us[k], color=colours[k])
plt.ylabel(r'$u$')
plt.xlabel(r'$k_3$')
plt.tight_layout()
plt.savefig(fig_save_path+'u_bif.pdf', dpi=300)

plt.figure()
for k in range(N):
    plt.plot(apis,ups[k], color=colours[k])
plt.ylabel(r'$v$')
plt.xlabel(r'$k_3$')
plt.tight_layout()
plt.savefig(fig_save_path+'up_bif.pdf', dpi=300)

plt.figure()
plt.plot(apis,coherences)
plt.ylabel(r'average phase-coherence')
plt.xlabel(r'$k_3$')
plt.ylim([-0.1,1.1])
plt.tight_layout()
plt.savefig('../plots/coh_bif.pdf', dpi=300)
plt.show()

print('we are done')

