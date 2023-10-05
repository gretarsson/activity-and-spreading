import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

L = 1.0
delta = 5.0
w1 = 5.0
w2 = 10.0
k0 = 1.0
k1 = 1.0
k2 = 1.0

def k3_crit(L,delta,w1,w2,k0,k1,k2,k):
    val = -L - 0.5*delta*(w1+w2) +  k0*k2/k1
    num = -((k+w1-w2)*(-4*delta**2*k0**2*k2**2*(k+w1-w2)*(k-w1+w2)-4*delta**2*k0*k1*k2*(k+w1-w2)*(k-w1+w2)*(2*L+delta*(w1+w2)) + k1**4*(2*L+delta*(w1+w2))**2 + 2*k1**3*(2*L+delta*(w1+w2))**3 + k1**2*(-4*delta**2*k0*k2*(k+w1-w2)*(k-w1+w2)+(2*L+delta*(w1+w2))**4)))
    denom = math.sqrt(-k-w1+w2)*(2*L+k1+delta*(w1+w2))*k1
    k3_cr = val + 0.5*math.sqrt(num)/denom
    
    return k3_cr

ks = np.linspace(0,4.99,100)
k3s = np.empty((ks.size,))

for i,k in enumerate(ks):
    k3s[i] = k3_crit(L,delta,w1,w2,k0,k1,k2,k)

#plt.style.use('seaborn-muted')
#palette = sns.color_palette()
#plt.plot(ks,k3s, c=palette[2], label=r'$\tilde{k}_1^{(f)}$')
#plt.axhline(y=1, xmin=5/10, xmax=10/10, c=palette[0], label=r'$\tilde{k}_1^{(p)}$')  
#plt.axvline(x=5, ymin=0, ymax=10, c='black', label=r'$K=\Delta\omega$')  
#plt.xlim([-0.1,10.1])
#plt.ylim([0.9,1.2])
#plt.xlabel(r'$K$')
#plt.ylabel(r'$\tilde{k}_1$')
#plt.legend()
#
## Add text to each part of the plot
#plt.text(1.75, 1.125, 'Free-running healthy')
#plt.text(1.75, 0.95, 'Free-running toxic')
#plt.text(7.25, 1.125, 'Phase-locked healthy')
#plt.text(7.25, 0.95, 'Phase-locked toxic')

# 2nd try
# Set the style
sns.set_style('ticks')
palette = sns.color_palette()

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(ks, k3s, c=palette[0], label=r'$\tilde{k}_1^{(f)}$')
ax.axhline(y=1, xmin=5/10, xmax=10/10, c=palette[0], label=r'$\tilde{k}_1^{(p)}$')
ax.axvline(x=5, ymin=0, ymax=10, c=palette[3], label=r'$K=\Delta\omega$')
ax.set_xlim([-0.1, 10.1])
ax.set_ylim([0.9, 1.2])
ax.set_xlabel(r'$K$', fontsize=14)
ax.set_ylabel(r'$k_3$', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
#ax.legend(fontsize=12, loc='upper right')

# Add text to each part of the plot
ax.text(0.1, 1.11, r'$k_3^{crit}$', fontsize=12)
ax.text(5.01, 1.01, r'$k_3 = \frac{k_0 k_2}{k_1}$', fontsize=12)
ax.text(4.7, 1.17, r'$K=|\Delta\omega|$', fontsize=12, rotation=90)

ax.text(1.04, 1.140, 'Free-running healthy', fontsize=14)
ax.text(1.14, 0.975, 'Free-running toxic', fontsize=14)
ax.text(6.3, 1.1, 'Phase-locked healthy', fontsize=14)
ax.text(6.4, 0.95, 'Phase-locked toxic', fontsize=14)

ax.text(5.2, 1.07, 'Symmetric', fontsize=11, rotation=90)
ax.text(4.6, 1.069, 'Asymmetric', fontsize=11, rotation=90)
ax.text(5.2, 0.93, 'Symmetric', fontsize=11, rotation=90)
ax.text(4.60, 0.929, 'Asymmetric', fontsize=11, rotation=90)

# Remove spines on top and right
sns.despine()

# save figure
plt.savefig('../plots/transcritical_bifurcation.pdf', dpi=300)
plt.show()
    
