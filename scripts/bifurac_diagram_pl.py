import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import seaborn as sns
import networkx as nx
from math import pi
from numba import jit
from feedback_helpers import feedback, compute_phase_coherence, plot_feedback, create_clustered_network
from tqdm import tqdm
import pickle
from tqdm import tqdm
sns_colors = sns.color_palette()

# --------------------------------------------------------------
# Here we plot the bifurcation diagrams for the two-node system
# dut = u2t - u1t
# --------------------------------------------------------------

# main parameters
run = True

# system parameters
K = 2
dw = 1
c = 1

# continuing parameter
ds = 4
M = 1000
dut_pos = np.linspace((K-dw)/c, (K-dw)/c + ds, M)
dut_neg = np.linspace((-K-dw)/c, (-K-dw)/c - ds, M)

# numerical integration
t_span = (0, 1000)
sf = 50
t_eval = np.linspace(t_span[0], t_span[1], 50*(t_span[1]-t_span[0])) 
y0 = [0.1]
method = 'RK45'
atol = 1e-8
rtol = 1e-6

# Create the plot
fig, ax = plt.subplots()
ax.set_xlim([(-K-dw)/c - ds, (K-dw)/c + ds])


def dphi_dt(t, y, K, dw, c, dut):
    phi = y[0]
    return dw + c*dut - K * np.sin(phi)
def rhs_phi(phi, K, dw, c, dut):
    return dw + c*dut - K * np.sin(phi)
def transform_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

phi_pos = [ [] for _ in range(M) ]
bin_size = 2*pi / 500
binss_pos = []
binss_neg = []
if run:
    for i, dut in enumerate(tqdm(dut_pos)):
        # numerically integrate phi(t)
        sol = solve_ivp(dphi_dt, t_span, y0, atol=atol, rtol=rtol, args=(K,dw,c,dut), t_eval=t_eval) 
        phii = np.mod(sol['y'][0], 2*pi)
        t = sol['t']

        # calculate bins of phi variable
        counts, bins = np.histogram(phii, bins=np.arange(0, 2*pi+bin_size, bin_size))

        # Create a scatter plot of the binned data
        colors = counts / np.max(counts) # Normalize the counts to [0, 1]

        # save the bin
        binss_pos.append((bins[:-1], dut, colors))

    # write binss
    with open('../simulations/binss_pos.pkl', 'wb') as f:
        # Use pickle to dump the variable binss to the file
        pickle.dump(binss_pos, f)

    for i, dut in enumerate(tqdm(dut_neg)):
        # numerically integrate phi(t)
        #y0 = np.random.uniform(low=0, high=6.28, size=1)
        sol = solve_ivp(dphi_dt, t_span, y0, atol=atol, rtol=rtol, args=(K,dw,c,dut), t_eval=t_eval) 
        phii = np.mod(sol['y'][0], 2*pi)
        t = sol['t']

        # calculate bins of phi variable
        counts, bins = np.histogram(phii, bins=np.arange(0, 2*pi+bin_size, bin_size))

        # Create a scatter plot of the binned data
        colors = counts / np.max(counts) # Normalize the counts to [0, 1]

        # save the bin
        binss_neg.append((bins[:-1], dut, colors))

    # write binss
    with open('../simulations/binss_neg.pkl', 'wb') as f:
        # Use pickle to dump the variable binss to the file
        pickle.dump(binss_neg, f)

# read binss
with open('../simulations/binss_pos.pkl', 'rb') as f:
    # Use pickle to load the variable binss from the file
    binss_pos = pickle.load(f)
with open('../simulations/binss_neg.pkl', 'rb') as f:
    # Use pickle to load the variable binss from the file
    binss_neg = pickle.load(f)


# solve fixed point solutions left branch)
x0 = 2*pi - pi/2
duts1 = np.linspace((-K-dw)/c,(K-dw)/c,2000)
fixed1 = []
for dut in duts1:
    sol = fsolve(rhs_phi, x0, args=(K,dw,c,dut))    
    fixed1.append(sol)
    x0 = sol

# solve fixed point solutions left branch)
x0 = pi/2
duts2 = np.flip(np.linspace((-K-dw)/c,(K-dw)/c,2000))
fixed2 = []
for dut in duts2:
    sol = fsolve(rhs_phi, x0, args=(K,dw,c,dut))    
    fixed2.append(sol)
    x0 = sol

# plot binss
jump=5
for (bins, dut, colors) in binss_pos:
    colors = [colors[i] for i in range(0,len(colors),jump)]
    ax.scatter([dut for _ in range(0,bins.size,jump)], [transform_pi(bins)[i] for i in range(0,bins.size,jump)], c=colors, cmap='Greys', s=15, alpha=0.5)
for (bins, dut, colors) in binss_neg:                            
    colors = [colors[i] for i in range(0,len(colors),jump)]
    ax.scatter([dut for _ in range(0,bins.size,jump)], [transform_pi(bins)[i] for i in range(0,bins.size,jump)], c=colors, cmap='Greys', s=15, alpha=0.5)

# transform fixed points to (-pi,pi)
fixed1 = np.array(fixed1)
fixed2 = np.array(fixed2)
fixed1 = np.mod(fixed1, 2*pi)
fixed2 = np.mod(fixed2, 2*pi)
fixed1 = transform_pi(fixed1)
fixed2 = transform_pi(fixed2)

# combine the branches (for plotting)
duts = np.concatenate((duts1,duts2)).reshape(-1)
fixed = np.concatenate((fixed1,fixed2)).reshape(-1)
sort_ind = np.argsort(fixed)
duts = duts[sort_ind]
fixed = fixed[sort_ind]

# the saddle nodes
SN1 = transform_pi(2*pi-pi/2)
SN2 = transform_pi(pi/2)

# slow manifold fixed point
SM1 = transform_pi(fsolve(rhs_phi, pi/4, args=(K,dw,c,0)))
SM2 = transform_pi(fsolve(rhs_phi, pi/1.5, args=(K,dw,c,0))) 

# find unstable and stable branches
stable_inds = np.where((SN1 < fixed) & (fixed < SN2))
unstable_inds1 = np.where((SN1 > fixed))
unstable_inds2 = np.where((SN2 < fixed))

# plot fixed point branches
ax.plot(duts[stable_inds],fixed[stable_inds],c='black')
ax.plot(duts[unstable_inds1],fixed[unstable_inds1],c='black',linestyle='--')
ax.plot(duts[unstable_inds2],fixed[unstable_inds2],c='black',linestyle='--')

# plot dots for saddle-node bifurcations
ax.scatter((-K-dw)/c, SN1, c='black', s=100)
ax.scatter((K-dw)/c, SN2, c='black', s=100)

# plot slow manifold fixed points
ax.scatter(0, SM1, c='grey', s=100)
ax.scatter(0, SM2, edgecolors='grey', s=100, marker='o', facecolors='none')

# plotting settings
ax.set_ylim([-pi,pi])
fontsize=18
# set custom y-ticks
yticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
yticklabels = ['-π', '-π/2', '0', 'π/2', 'π']
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
# set labels
ax.set_ylabel(f'$\phi(t)$', fontsize=fontsize)
ax.set_xlabel(f'$\Delta v$', fontsize=fontsize)
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

# simulate the whole-system and plot evolution on figure
# model parameters
rho=1e-3;a0=1.0;ai=1;aii=1;api=0.75
eps=2*1e-1;K=2.;c=1;delta=1*eps;w=np.array([2,1]).astype(np.float64)
W = np.array([[0,1],[1,0]]).astype(np.float64)
L = np.array([[1,-1],[-1,1]]).astype(np.float64)
N = 2

# solver settings
t_span=(0,15);method='BDF';atol=1e-8;rtol=1e-6;max_step=1e-4;t_eval_sim=np.linspace(t_span[0], t_span[1], int(1000/eps))
y0=[1.0,1.0, 0.1, 5.0, 0.2, 1.15]


# solve system
print(f'solving system...')
sol = solve_ivp(lambda t, y: feedback(t, y, L, W, rho, a0, ai, aii, api, eps, delta, K, c, w), \
                         t_span, y0, method=method, atol=atol, rtol=rtol, max_step=max_step, t_eval=t_eval_sim)

# extract and store steady-state
u = sol.y[0:N]
up = sol.y[N:2*N]
dup = up[1] - up[0]
theta = sol.y[2*N:3*N,:]
dtheta = transform_pi( transform_pi(theta[0]) - transform_pi(theta[1]) )
t = sol.t


# plot the darn thing
colors = np.linspace(0,1,dup.size)
plt.scatter(dup,dtheta,color=sns_colors[3], alpha=0.25, s=5)

# REPEAT TOXIC
# simulate the whole-system and plot evolution on figure
# model parameters
rho=1e-3;a0=1.0;ai=1;aii=1;api=1.25
eps=0.75*1e-1;K=2.;c=1;delta=1*eps;w=np.array([2,1]).astype(np.float64)
W = np.array([[0,1],[1,0]]).astype(np.float64)
L = np.array([[1,-1],[-1,1]]).astype(np.float64)
N = 2

# solver settings
t_span=(0,15);method='BDF';atol=1e-8;rtol=1e-6;max_step=1e-4;t_eval_sim=np.linspace(t_span[0], t_span[1], int(1000/eps))
y0=[1.0,1.0, 7.0, 0.1, 0.2, 1.15]


# solve system
print(f'solving system...')
sol = solve_ivp(lambda t, y: feedback(t, y, L, W, rho, a0, ai, aii, api, eps, delta, K, c, w), \
                         t_span, y0, method=method, atol=atol, rtol=rtol, max_step=max_step, t_eval=t_eval_sim)

# extract and store steady-state
u = sol.y[0:N]
up = sol.y[N:2*N]
dup = up[1] - up[0]
theta = sol.y[2*N:3*N,:]
dtheta = transform_pi( transform_pi(theta[0]) - transform_pi(theta[1]) )
t = sol.t


# plot the darn thing
colors = np.linspace(0,1,dup.size)
plt.title(f'$K\geq|\Delta\omega|$')
plt.scatter(dup,dtheta,color=sns_colors[0], alpha=0.25, s=5)

# vertical line at 0
plt.axvline(x=0, color=sns_colors[1], alpha=1.0, linestyle='--', linewidth=3)

# we're done
plt.tight_layout()
plt.savefig('../plots/locked_regime.pdf', dpi=150)
plt.savefig('../plots/locked_regime.png')

