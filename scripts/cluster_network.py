import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
import networkx as nx
from math import pi
from numba import jit
from feedback_helpers import feedback, compute_phase_coherence, plot_feedback, create_clustered_network
from tqdm import tqdm
import pickle

# ---------------------------------
# Here we simulate a one-species
# heterodimer model
# -----------------------------------
np.random.seed(0)
run = True
# heterodimer parameters
n=10;M=3;m=2
N=n*M
rho=0.5;l=0.1;a0=1.0;ai=1;aii=1;api=0.75
eps=0.001;delta=1*eps;K=10.0/N;c=0.95
w1 = np.random.normal(loc=5.0, scale=0.5,size=n)
w2 = np.random.normal(loc=10.0,scale=0.5,size=n)
w3 = np.random.normal(loc=15.0,scale=0.5,size=n)
w = np.concatenate((w1,w2,w3))

# solver settings
t_span=(0,50);method='BDF';atol=1e-8;rtol=1e-6;max_step=1e-4

# initial conditions
y0_u = np.random.normal(loc=1.0,scale=0.0,size=N)
y0_up = np.random.normal(loc=1e-2,scale=0.00,size=N)
y0_theta = np.random.normal(loc=0.5,scale=0.25,size=N)
y0 = np.concatenate([y0_u, y0_up, y0_theta])

# Create an empty chain network
G = create_clustered_network(n,M,m)
L = nx.laplacian_matrix(G).toarray().astype(np.float64)
A = nx.adjacency_matrix(G).toarray().astype(np.float64)

if run:
    # solve system
    sol = solve_ivp(lambda t, y: feedback(t, y, L, A, rho, a0, ai, aii, api, eps, delta, K, c, w), \
                             t_span, y0, method=method, atol=atol, rtol=rtol, max_step=max_step)
    
    # save sol
    with open('../simulations/cluster.pkl', 'wb') as f:
        pickle.dump(sol, f)

# load sol
with open('../simulations/cluster.pkl', 'rb') as f:
    sol = pickle.load(f)

# Get the seaborn muted color palette
palette = sns.color_palette("muted")
blue = pltc.to_hex(palette[0])
red = pltc.to_hex(palette[3])
green = pltc.to_hex(palette[2])
colours = []
for _ in range(n):
    colours.append(blue)
for _ in range(n):
    colours.append(red)
for _ in range(n):
    colours.append(green)
colours.append(pltc.to_hex(palette[-1]))

plt.figure()
nx.draw(G, node_color=colours[0:30])

# plot and save figures
plt.figure()
plot_feedback(sol, '../plots/cluster_', A, eps, delta, K, c, w, colours=colours)
plt.show()
print('we are done')

