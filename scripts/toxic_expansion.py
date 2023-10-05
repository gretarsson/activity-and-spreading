import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# --------------------------------------------------------
# Here we simulate a one-species
# heterodimer model and compare steady-states
# of simulations to expansions of the toxic fixed points
# --------------------------------------------------------

run = True
fig_save_path = '../plots/'
# heterodimer parameters
rho=1; a0=1; ai=1; aii=1; api=0.25
alpha=0; L_val=1
n = 30
N = 2
alphas = np.linspace(0,0.25,n)

# solver settings
#t_span = (0,7500)  # high accuracy in steady-state
t_span = (0,100)
y0 = (1, 1, 0.001, 0.001)
method = 'RK45'
atol = 10**-8
rtol = 10**-6

fixed_us = np.zeros((N,n))
fixed_ups = np.zeros((N,n))
if run:
    for k, alpha in enumerate(alphas):
        # network parameters
        L = np.array([[L_val,-L_val],[-L_val,L_val]])
        Malpha = np.array([[alpha, 0], [-alpha, 0]])

        # construct laplacian, list of edges, and list of neighours
        neighbours = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if L[i,j] < 0:
                    neighbours[i].append(j)
                    neighbours[j].append(i)

        # spreading dynamics
        def rhs(t, y):
            # set up variables as lists indexed by node k
            u = np.array([y[i] for i in range(N)])
            up = np.array([y[i] for i in range(N, 2*N)])

            # scale Laplacian by diffusion constant
            global L
            L = rho*L
            
            # nodal dynamics
            du, dup = [[] for _ in range(2)]
            for k in range(N):
                # index list of node k and its neighbours
                neighbours_k = neighbours[k] + [k]

                # heterodimer dynamics
                duk = sum([-(L[k,l] + Malpha[k,l])*u[l] for l in neighbours_k]) + a0 - ai*u[k] - aii*u[k]*up[k]
                dupk = sum([-(L[k,l] + Malpha[k,l])*up[l] for l in neighbours_k]) - api*up[k] + aii*u[k]*up[k]

                ## append
                du.append(duk)
                dup.append(dupk)

            # pack right-hand side
            rhs = [*du, *dup]

            return rhs 

        sol = solve_ivp(rhs, t_span, y0, method=method, atol=atol, rtol=rtol)

        u = sol.y[0:N,:]
        up = sol.y[N:2*N,:]
        t = sol.t

        fix_u = u[:,-1]
        fix_up = up[:,-1]
        fixed_us[:,k] = fix_u
        fixed_ups[:,k] = fix_up

    # Save fixed_us
    with open('../simulations/fixed_us.pkl', 'wb') as f:
        pickle.dump(fixed_us, f)
    # Save fixed_ups
    with open('../simulations/fixed_vs.pkl', 'wb') as f:
        pickle.dump(fixed_ups, f)
    # Save alphas
    with open('../simulations/alphas.pkl', 'wb') as f:
        pickle.dump(alphas, f)

# Load fixed_us
with open('../simulations/fixed_us.pkl', 'rb') as f:
    fixed_us = pickle.load(f)
# Load fixed_ups
with open('../simulations/fixed_vs.pkl', 'rb') as f:
    fixed_ups = pickle.load(f)
# Load alphas
with open('../simulations/alphas.pkl', 'rb') as f:
    alphas = pickle.load(f)

# plot dependency on alpha
fig_u, ax_u = plt.subplots()
fig_up, ax_up = plt.subplots()
plt.style.use('seaborn-muted')
colors = sns.color_palette("hls", 8)
blue = colors[-3]
red = colors[0]
green = colors[2]
colours= [blue,red]
for i in range(N):
    ax_u.plot(alphas, fixed_us[i,:], c=colours[i])

    ax_up.plot(alphas, fixed_ups[i,:], c=colours[i])


# taylor expansions
pred_u1 = api/aii + (alphas * api * (a0 * aii - 2 * L_val * api - ai * api)) / (aii * (2 * L_val * a0 * aii + 4 * L_val**2 * api + a0 * aii * api - ai * api**2))
pred_up1 = (a0 * aii - ai * api) / (aii * api) - (alphas * (a0 * aii - ai * api) * (a0 * aii + 2 * L_val * api + api**2)) / (aii * api * (2 * L_val * a0 * aii + 4 * L_val**2 * api + a0 * aii * api - ai * api**2))
pred_u2 = api/aii - (alphas * api * (a0 * aii - 2 * L_val * api - ai * api)) / (aii * (2 * L_val * a0 * aii + 4 * L_val**2 * api + a0 * aii * api - ai * api**2))
pred_up2 = (a0 * aii - ai * api) / (aii * api) + (alphas * (a0 * aii - ai * api) * (a0 * aii + 2 * L_val * api + api**2)) / (aii * api * (2 * L_val * a0 * aii + 4 * L_val**2 * api + a0 * aii * api - ai * api**2))

# plot analytic approximations
ax_u.plot(alphas, pred_u1, c=colours[0], linestyle='dashdot')
ax_u.plot(alphas, pred_u2, c=colours[1], linestyle='dashdot')

ax_up.plot(alphas, pred_up1, c=colours[0], linestyle='dashdot')
ax_up.plot(alphas, pred_up2, c=colours[1], linestyle='dashdot')

# show
ax_u.set_ylabel(r'Steady-states of $u_1$ and $u_2$')
ax_u.set_xlabel(r'$A$')
ax_u.set_xlim([0,0.25])
ax_up.set_ylabel(r'Steady-states of $v_1$ and $v_2$')
ax_up.set_xlabel(f'$A$')
ax_up.set_xlim([0,0.25])
fig_u.savefig(fig_save_path + 'ss_healthy_expansion.pdf', dpi=300)
fig_up.savefig(fig_save_path + 'ss_toxic_expansion.pdf', dpi=300)
plt.show()

