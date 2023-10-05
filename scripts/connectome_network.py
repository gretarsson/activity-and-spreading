import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
import networkx as nx
from math import pi
from numba import jit
from feedback_helpers import feedback, compute_phase_coherence, plot_feedback, create_clustered_network, \
                                get_column_index, read_ods_file, average_labels2, average_labels
from tqdm import tqdm
import csv
import pickle
colours = sns.color_palette('muted')

# ---------------------------------
# Here we simulate a one-species
# heterodimer model
# -----------------------------------
fig_save_path = '../plots/connectome_'
np.random.seed(1)
run = True
# heterodimer parameters
N=83;
rho=1e-3;a0=1.0;ai=1;aii=1;api=0.9
eps=0.01;K=0.1;c=10
w = np.random.normal(loc=10.0, scale=0.5,size=N)
t_cut = 10

# changing parameter
M = 1  # 20
deltas = np.linspace(0.0,2,M)*eps

# solver settings tspan 250
t_span=(0,250);method='BDF';atol=1e-6;rtol=1e-3;max_step=1e-3

# initial conditions
y0_u = np.random.normal(loc=1.0,scale=0.0,size=N)
y0_up = np.random.normal(loc=0,scale=0.00,size=N)
y0_up[26] = 1e-2  # left and right entorhinal cortex
y0_up[67] = 1e-2
y0_theta = np.random.normal(loc=0.5,scale=0.25,size=N)
y0 = np.concatenate([y0_u, y0_up, y0_theta])

# Read Bick & Goriely's coupling matrix
W = np.zeros((N,N))
with open('../data/budapest_connectome.csv', 'r') as f:
    W = list(csv.reader(f, delimiter=','))
W = np.array(W, dtype=np.float64)
W = np.maximum( W, W.transpose() )

# Create network matrices
G = nx.from_numpy_matrix(W)
L = nx.laplacian_matrix(G).toarray().astype(np.float64)

# initialize steady-state vectors
us = np.empty((N,M))
ups = np.empty((N,M))
first_us = np.empty((N,M))
first_ups = np.empty((N,M))
avg_freqs = np.empty((N,M))
coherences = np.empty((M,))
if run:
    for m,delta in enumerate(tqdm(deltas)):
        # solve system
        sol = solve_ivp(lambda t, y: feedback(t, y, L, W, rho, a0, ai, aii, api, eps, delta, K, c, w), \
                                 t_span, y0, method=method, atol=atol, rtol=rtol, max_step=max_step)
        
        # extract and store steady-state
        u = sol.y[0:N]
        up = sol.y[N:2*N]
        thetas = sol.y[2*N:3*N,:]
        t = sol.t
        subinds = np.where(t > t[-1]-t_cut)
        thetas = thetas[:,subinds[0]]
        up_sub = up[:,subinds[0]]
        t_sub = t[subinds]
        dthetas = []
        for i in range(t_sub.size-1):
            theta = thetas[:,i]
            dtheta = w - c*up_sub[:,i] + K/N * np.diag(np.dot(np.sin(theta - np.expand_dims(theta, axis=1)), W))  # without w
            dtheta = (t_sub[i+1] - t_sub[i])*dtheta 
            dthetas.append(dtheta)
        dthetas = np.array(dthetas)

        u_thr = 0.95
        up_thr = 0.01
        first_u = t[get_column_index(u, u_thr, is_larger=False)]
        first_up = t[get_column_index(up, up_thr, is_larger=True)]
        first_us[:,m] = first_u
        first_ups[:,m] = first_up

        us[:,m] = u[:,-1]
        ups[:,m] = up[:,-1]
        avg_freqs[:,m] = 1/(t_sub[-1]-t_sub[0]) * np.sum(dthetas,axis=0) 
        coherences[m] = np.mean(compute_phase_coherence(theta))

        # plot and save figures
        colours = sns.color_palette('hls',N)
        plot_feedback(sol, fig_save_path, W, eps, delta, K, c, w, colours=colours)

    # save store
    with open('../simulations/connetome_sim.pkl', 'wb') as f:
        pickle.dump((us, ups, coherences, first_us, first_ups, avg_freqs), f)


# load store
with open('../simulations/connetome_sim.pkl', 'rb') as f:
    us, ups, coherences, first_us, first_ups, avg_freqs = pickle.load(f)

# load braak staging and find averages of braak regions
braak = read_ods_file('../data/braak_regions_twocol.ods')
braak = braak[~(braak[:,1] == 'N'),:]
braak_labels = np.unique(braak[:,1])
L = len(braak_labels)
braak_first_us = average_labels2(first_us, braak)
braak_first_ups = average_labels2(first_ups, braak)
braak_us = average_labels2(us, braak)
braak_ups = average_labels2(ups, braak)
braak_freqs = average_labels2(avg_freqs, braak)

# convert deltas    
deltas = deltas / eps
    
# plot steady=state
plt.style.use('seaborn-muted')
plt.figure()
for l in range(L):
    plt.plot(deltas,braak_us[l], label=braak_labels[l])
plt.ylabel(r'$u$')
plt.xlabel(r'$\delta$')
plt.legend()
plt.savefig(fig_save_path+'u_bif.pdf', dpi=300)

plt.figure()
for l in range(L):
    plt.plot(deltas,braak_ups[l], label=braak_labels[l])
plt.ylabel(r'$v$')
plt.xlabel(r'$\delta$')
plt.legend()
plt.savefig(fig_save_path+'v_bif.pdf', dpi=300)

plt.figure()
for l in range(L):
    plt.plot(deltas,braak_first_us[l], label=braak_labels[l])
plt.ylabel(r'$u_T$')
plt.xlabel(r'$\delta$')
plt.legend()
plt.savefig(fig_save_path+'u_time.pdf', dpi=300)
plt.figure()

for l in range(L):
    plt.plot(deltas,braak_first_ups[l])
plt.ylabel(r'$v_T$')
plt.xlabel(r'$\delta$')
plt.legend()
plt.savefig(fig_save_path+'v_time.pdf', dpi=300)


plt.figure()
for l in range(L):
    plt.plot(deltas,braak_freqs[l])
plt.ylabel(r'Average frequencies')
plt.xlabel(r'$\delta$')
plt.savefig(fig_save_path+'avg_freqs.pdf', dpi=300)


plt.figure()
plt.plot(deltas,coherences)
plt.ylabel(r'Average phase-coherence')
plt.xlabel(r'$\delta$')
plt.ylim([-0.1,1.1])
plt.savefig(fig_save_path+'coh_bif.pdf', dpi=300)

print('we are done')

