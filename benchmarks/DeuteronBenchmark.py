# This compares FCI and CCSD for a two-body interaction from chiral EFT 
__authors__   =  "Thomas Papenbrock"
__credits__   =  ["Thomas Papenbrock"]
__copyright__ = "(c) Thomas Papenbrock"
__license__   = "BSD-3-Clause"
__date__      = "2025-09-02"


import pathlib

import numpy as np
from opt_einsum import contract
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh as arpack_eigsh
from NuLattice.cpu.fci.few_body_diagonalization import fill_1b_op_in_2b_basis, fill_2b_op_in_2b_basis
from NuLattice.cpu.ccm.ccDgrams import pAB, pIJ

## exact diagonalization starts here for two-body problem
# read interaction

this_directory = pathlib.Path(__file__).parent

file0=this_directory / 'H2_data/deuteron_Hvac_0B.dat'
file1=this_directory / 'H2_data/deuteron_Hvac_1B.dat'
file2=this_directory / 'H2_data/deuteron_Hvac_2B.dat'

data0=np.loadtxt(file0)


data1=np.loadtxt(file1)
nstat=0
for line in data1:
    if line[0] > nstat: 
        nstat=line[0]
    if line[1] > nstat: 
        nstat=line[1]

nstat=int(nstat)+1 #counting started at 0
print("number of single-particle states:", nstat)

Fvals=[]
for line in data1:
    i = int(line[0])
    j = int(line[1])
    val = line[2]
    Fvals.append([i,j,val])
    
data2=np.loadtxt(file2)
Vvals=[]
for line in data2:
    i = int(line[0])
    j = int(line[1])
    k = int(line[2])
    l = int(line[3])
    val = line[4]
    if i < j and k < l:
        Vvals.append([i, j, k, l, val])

# build two-body basis
twobody_bas=[]
for i in range(nstat): 
    for j in range(i+1,nstat): 
        twobody_bas.append((i,j))

                         
dim2 = len(twobody_bas)
print("number of deuteron states:", dim2)
                                   
vals     = range(dim2)
lookup2b = dict(zip(twobody_bas,vals))
del twobody_bas  # Deallocate
print("done")

## construct matrices for two-body system

T2_dat, T2_row, T2_col = fill_1b_op_in_2b_basis(lookup2b,Fvals,nstat)
    
print("number of one-body-operator matrix elements in two-body system:", len(T2_row))

## pack into compressed storage row matrix
T2_mat=sparse.csr_matrix( (T2_dat, (T2_row, T2_col)), shape=(dim2,dim2) )
del T2_dat, T2_row, T2_col

V2_dat, V2_row, V2_col = fill_2b_op_in_2b_basis(lookup2b,Vvals)

print("number of two-body-operator matrix elements in two-body system:", len(V2_row))
    
V2_mat=sparse.csr_matrix( (V2_dat, (V2_row, V2_col)), shape=(dim2,dim2) )
del V2_dat, V2_row, V2_col

# Diagonalize deuteron 
H2_mat = T2_mat + V2_mat
vals2, vecs2 = arpack_eigsh(H2_mat, k=4, which='SA')
print("smallest EVs:", vals2)

#Assert correctness
message = "Spin-1 state needs to have a three-fold degenerate ground state"
ndig_fci=9
assert (round(vals2[0],ndig_fci) == round(vals2[1],ndig_fci) and 
        round(vals2[0],ndig_fci) == round(vals2[2],ndig_fci) ), message


## coupled cluster starts here
# the T1 and T2 equations are coded diagram by diagram as they appear in Crawford and Schaefer

def H1ccsd(f_pp, f_ph, f_hh, v_ppph, v_pphh, v_phph, v_phhh, t1, t2):  
    h1 = np.zeros_like(t1)
    h1 +=  f_ph
    h1 +=  contract("ac,ci->ai", f_pp, t1)
    h1 += -contract("ki,ak->ai", f_hh, t1)
    h1 += -contract("akci,ck->ai", v_phph, t1)
    h1 +=  contract("ck,acik->ai", f_ph, t2)
    h1 += -contract("cdak,cdki->ai", v_ppph, t2)*0.5
    h1 += -contract("cikl,cakl->ai", v_phhh, t2)*0.5
    h1 += -contract("ck,ci,ak->ai", f_ph, t1, t1, optimize='greedy')
    h1 += -contract("cikl,ck,al->ai", v_phhh, t1, t1, optimize='greedy')
    h1 += -contract("cdak,ck,di->ai", v_ppph, t1, t1, optimize='greedy')
    h1 += -contract("cdkl,ck,di,al->ai", v_pphh, t1, t1, t1, optimize='greedy')
    h1 +=  contract("cdkl,ck,dali->ai", v_pphh, t1, t2, optimize='greedy')
    h1 += -contract("cdkl,cdki,al->ai",v_pphh, t2, t1, optimize='greedy')*0.5 
    h1 += -contract("cdkl,cakl,di->ai",v_pphh, t2, t1, optimize='greedy')*0.5  
    return h1

def H2ccsd(f_pp, f_ph, f_hh, v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh, t1, t2):
    h2 = np.zeros_like(t2)
    h2 += v_pphh
    h2 +=  pAB( contract("bc,acij->abij", f_pp, t2) )
    h2 += -pIJ( contract("kj,abik->abij", f_hh, t2) )
    h2 += contract("klij,abkl->abij", v_hhhh, t2)*0.5
    h2 += contract("abcd,cdij->abij", v_pppp, t2)*0.5
    h2 += -pAB( pIJ( contract("bkcj,acik->abij", v_phph, t2) ) )
    h2 += pIJ( contract("abcj,ci->abij", v_ppph, t1) )
    h2 += pAB( contract("bkij,ak->abij", v_phhh, t1) )
    h2 += pIJ( pAB( contract("cdkl,acik,dblj->abij", v_pphh, t2, t2, optimize="greedy") ) )*0.5
    h2 += contract("cdkl,cdij,abkl->abij", v_pphh, t2, t2, optimize="greedy")*0.25
    h2 += -pAB( contract("cdkl,acij,bdkl->abij", v_pphh, t2, t2, optimize="greedy") )*0.5
    h2 += -pIJ( contract("cdkl,abik,cdjl->abij", v_pphh, t2, t2, optimize="greedy") )*0.5
    h2 += pAB( contract("klij,ak,bl->abij", v_hhhh, t1, t1, optimize="greedy") )*0.5
    h2 += pIJ( contract("abcd,ci,dj->abij", v_pppp, t1, t1, optimize="greedy") )*0.5
    h2 += -pIJ( pAB( np.einsum("bkci,ak,cj->abij", v_phph, t1, t1, optimize="greedy") ) )
    h2 += pAB( contract("ck,ak,bcij->abij", f_ph, t1, t2, optimize="greedy") )
    h2 += pIJ( contract("ck,ci,abjk->abij", f_ph, t1, t2, optimize="greedy") )
    h2 += -pIJ( contract("cikl,ck,ablj->abij", v_phhh, t1, t2, optimize="greedy") )
    h2 += -pAB( contract("cdak,ck,dbij->abij", v_ppph, t1, t2, optimize="greedy") )
    h2 += pIJ( pAB( contract("dcak,di,bcjk->abij", v_ppph, t1, t2, optimize="greedy") ) )
    h2 += -pIJ( pAB( contract("cikl,al,bcjk->abij", v_phhh, t1, t2, optimize="greedy") ) )
    h2 += pIJ( contract("cjkl,ci,abkl->abij", v_phhh, t1, t2, optimize="greedy") )*0.5
    h2 += pAB( contract("cdbk,ak,cdij->abij", v_ppph, t1, t2, optimize="greedy") )*0.5
    h2 += pIJ( pAB( contract("cdbk,ci,ak,dj->abij", v_ppph, t1, t1, t1, optimize="greedy") ) )*0.5
    h2 += pIJ( pAB( contract("cjkl,ci,ak,bl->abij", v_phhh, t1, t1, t1, optimize="greedy") ) )*0.5
    h2 += -pIJ( contract("cdkl,ck,di,ablj->abij", v_pphh, t1, t1, t2, optimize="greedy") )
    h2 += -pAB( contract("cdkl,ck,al,dbij->abij", v_pphh, t1, t1, t2, optimize="greedy") )
    h2 += pIJ( contract("cdkl,ci,dj,abkl->abij", v_pphh, t1, t1, t2, optimize="greedy") )*0.25
    h2 += pAB( contract("cdkl,ak,bl,cdij->abij", v_pphh, t1, t1, t2, optimize="greedy") )*0.25
    h2 += pIJ( pAB( contract("cdkl,ci,bl,adkj->abij", v_pphh, t1, t1, t2, optimize="greedy") ) )
    h2 += pIJ( pAB( contract("cdkl,ci,ak,dj,bl->abij", v_pphh, t1, t1, t1, t1, optimize="greedy") ) )*0.25    
    return h2 


    
def eCCSD(t1,t2,f_ph,v_pphh):
    res = 0.0
    res += contract("ai,ai",f_ph,t1)
    res += contract("abij,abij",v_pphh,t2)*0.25
    res += contract("abij,ai,bj",v_pphh,t1,t1,optimize="greedy")*0.5
    return res

def init_t(f_pp,f_ph,f_hh,v_pphh,triples=False):
    fp =  np.diagonal(f_pp)
    fh = -np.diagonal(f_hh)
    denom1 = -np.add.outer(fp,fh)
    t1 = f_ph/denom1
    
    dh = np.add.outer(fh,fh)
    dp = np.add.outer(fp,fp)
    denom2 = -np.add.outer(dp,dh)
    t2 = v_pphh/denom2
    
    return t1, t2, denom1, denom2

# read and process normal-ordered matrix elements
fileNO0=this_directory / 'H2_data/deuteron_HNO_basis1_0B.dat'
fileNO1=this_directory / 'H2_data/deuteron_HNO_basis1_1B.dat'
fileNO2=this_directory / 'H2_data/deuteron_HNO_basis1_2B.dat'

hnum=2
hole=(0,2)

Eref=np.loadtxt(fileNO0)
print("Eref=", Eref)

data1=np.loadtxt(fileNO1)
nstat=0
for line in data1:
    if line[0] > nstat: 
        nstat=line[0]
    if line[1] > nstat: 
        nstat=line[1]

nstat=int(nstat)+1 #counting started at 0

stats=tuple(range(nstat))
part = tuple(set(stats)-set(hole))


pnum=nstat-hnum
f_pp = np.zeros((pnum,pnum))
f_ph = np.zeros((pnum,hnum))
f_hh = np.zeros((hnum,hnum))

for line in data1:
    p = line[0]
    q = line[1]
    val = line[2]
    if p in hole:
        i = hole.index(p)
        if q in hole:
            j = hole.index(q)
            f_hh[i,j]=val
        else:
            a = part.index(q)
            f_ph[a,i]=val
    else:
        a = part.index(p)
        if q in hole:
            j = hole.index(q)
            f_ph[a,j]=val
        else:
            b = part.index(q)
            f_pp[a,b]=val
        

v_pppp=np.zeros((pnum,pnum,pnum,pnum))
v_ppph=np.zeros((pnum,pnum,pnum,hnum))
v_pphh=np.zeros((pnum,pnum,hnum,hnum))
v_phph=np.zeros((pnum,hnum,pnum,hnum))
v_phhh=np.zeros((pnum,hnum,hnum,hnum))
v_hhhh=np.zeros((hnum,hnum,hnum,hnum))

data2=np.loadtxt(fileNO2)
for line in data2:
    p = int(line[0])
    q = int(line[1])
    r = int(line[2])
    s = int(line[3])
    val = line[4]
    if p in part and q in part and r in part and s in part:
        a=part.index(p)
        b=part.index(q)
        c=part.index(r)
        d=part.index(s)
        v_pppp[a,b,c,d]=val
    elif p in part and q in part and r in part and s in hole:
        a=part.index(p)
        b=part.index(q)
        c=part.index(r)
        d=hole.index(s)
        v_ppph[a,b,c,d]=val
    elif p in part and q in part and r in hole and s in hole:
        a=part.index(p)
        b=part.index(q)
        c=hole.index(r)
        d=hole.index(s)
        v_pphh[a,b,c,d]=val
    elif p in part and q in hole and r in part and s in hole:
        a=part.index(p)
        b=hole.index(q)
        c=part.index(r)
        d=hole.index(s)
        v_phph[a,b,c,d]=val
    elif p in part and q in hole and r in hole and s in hole:
        a=part.index(p)
        b=hole.index(q)
        c=hole.index(r)
        d=hole.index(s)
        v_phhh[a,b,c,d]=val
    elif p in hole and q in hole and r in hole and s in hole:
        a=hole.index(p)
        b=hole.index(q)
        c=hole.index(r)
        d=hole.index(s)
        v_hhhh[a,b,c,d]=val


# Solve CCSD equations
t1, t2, denom1, denom2 = init_t(f_pp,f_ph,f_hh,v_pphh)
erg_old = eCCSD(t1,t2,f_ph,v_pphh)+Eref
print("initial energy:", erg_old)

iter = 1000
mix=0.6
eps = 1.e-9
for i in range(iter):
    if i < 10:
        gap=1.0
    else:
        gap=0.0
        
    t1_old = t1.copy()
    t2_old = t2.copy()
    
    h1_ph = H1ccsd(f_pp, f_ph, f_hh, v_ppph, v_pphh, v_phph, v_phhh, t1_old, t2_old) 
    t1 += h1_ph/(denom1+gap)
    
    h2_pphh = H2ccsd(f_pp, f_ph, f_hh, v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh, t1_old, t2_old)
    t2 += h2_pphh/(denom2+gap)
    
    t1 = mix*t1 + (1.0-mix)*t1_old
    t2 = mix*t2 + (1.0-mix)*t2_old
    erg_new = eCCSD(t1,t2,f_ph,v_pphh)+Eref
    diff = np.abs(erg_new-erg_old)
    print(i, erg_new, diff)
    erg_old = erg_new
    if diff < eps:
        print("success! Energy is ", erg_new)
        break


# Assert correctness
message = "CCSD different from FCI"
ndig=min(ndig_fci, round(-np.log10(eps)) - 2)  # number of digits based on eps
assert round(vals2[0],ndig) == round(erg_new,ndig), message
