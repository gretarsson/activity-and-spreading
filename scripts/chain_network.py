import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from math import pi
from numba import jit
from feedback_helpers import feedback, compute_phase_coherence, plot_feedback
from tqdm import tqdm
import pickle
colours = sns.color_palette('muted')

# ---------------------------------
# Here we simulate a one-species
# heterodimer model
# -----------------------------------
np.random.seed(0)
run = True
# heterodimer parameters
N=5
rho=0.5;l=0.1;a0=1.0;ai=1;aii=1;api=0.75
eps=0.001;delta=1*eps;K=1;c=0.5
w = np.arange(N) * 2.0 + 5

# solver settings
t_span=(0,80);method='BDF';atol=1e-6;rtol=1e-3;max_step=1e-4

# initial conditions
y0_u = np.random.normal(loc=1.0,scale=0.0,size=N)
y0_up = np.random.normal(loc=1e-4,scale=0.00,size=N)
y0_theta = np.random.normal(loc=0.5,scale=0.25,size=N)
y0 = np.concatenate([y0_u, y0_up, y0_theta])

# Create an empty chain network
G = nx.Graph()
for i in range(N):
    G.add_node(i)
for i in range(N - 1):
    G.add_edge(i, i+1, weight=l)
L = nx.laplacian_matrix(G).toarray().astype(np.float64)
A = nx.adjacency_matrix(G).toarray().astype(np.float64)
pos = {node: (node, 0) for node in G.nodes}
nx.draw(G,pos, node_color=colours[0:5])

if run:
    # solve system
    sol = solve_ivp(lambda t, y: feedback(t, y, L, A, rho, a0, ai, aii, api, eps, delta, K, c, w), \
                             t_span, y0, method=method, atol=atol, rtol=rtol, max_step=max_step)
    
    # save sol
    with open('../simulations/chain.pkl', 'wb') as f:
        pickle.dump(sol, f)

# load sol
with open('../simulations/chain.pkl', 'rb') as f:
    sol = pickle.load(f)

# plot and save figures
plot_feedback(sol, '../plots/chain_', A, eps, delta, K, c, w, colours=colours)
plt.show()
print('we are done')

