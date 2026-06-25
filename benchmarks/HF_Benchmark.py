import pathlib
import numpy as np
import NuLattice.cpu.hf.hartree_fock as hf

## HF check

E_HF_check = -20.3588628  #energy to compare with

this_directory = pathlib.Path(__file__).parent

file1=this_directory / "HF_data/ham_e2_1B.dat"
file2=this_directory / "HF_data/ham_e2_2B.dat"

data1=np.loadtxt(file1)
data2=np.loadtxt(file2)

nstat = 1 + round(max(data1[:,0]))
print("number of s.p. states:", nstat)
           
# make list of 1-body marix elements 
h1b = []
for line in data1:
    [p, q, val] = line
    h1b.append([round(p), round(q), val])

# make list of 2-body marix elements 
h2b = []
for line in data2:
    [p, q, r, s, val] = line
    if p < q and r < s:
        h2b.append([round(p), round(q), round(r), round(s), val])

# initialize density matrix
dens = np.zeros((nstat,nstat))
dens[0,0]=1.0
dens[1,1]=1.0
dens[2,2]=1.0
dens[3,3]=1.0


ndigit=7


# HF computation
eps=1.e-8
mix = 0.8
max_iter=100
verbose = True
h3b=[] #we have no 3-body force
erg, trafo, conv = hf.solve_HF(h1b, h2b, h3b, dens,
                               mix=mix, eps=eps, max_iter=max_iter, verbose=verbose)

if conv:
    print("HF energy (MeV) = ", erg)
    print("vs exact value  = ", E_HF_check)
else:
    print("HF did not converge")
    
assert np.abs(E_HF_check - round(erg,ndigit)) < 10.0**(-ndigit), "energies do not agree" 

