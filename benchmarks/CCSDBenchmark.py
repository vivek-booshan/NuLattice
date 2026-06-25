import pathlib
import NuLattice.cpu.ccm.coupled_cluster as ccm
import numpy as np

this_directory = pathlib.Path(__file__).parent

# read and process normal-ordered matrix elements
fileNO0=this_directory / 'H2_data/deuteron_HNO_basis1_0B.dat'
fileNO1=this_directory / 'H2_data/deuteron_HNO_basis1_1B.dat'
fileNO2=this_directory / 'H2_data/deuteron_HNO_basis1_2B.dat'

hnum=2
hole=(0,2)

Eref=np.loadtxt(fileNO0)

data1=np.loadtxt(fileNO1)
nstat=0
for line in data1:
    if line[0] > nstat: 
        nstat=line[0]
    if line[1] > nstat: 
        nstat=line[1]

nstat=int(nstat)+1

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
v_pppp_sparse = []
v_ppph=np.zeros((pnum,pnum,pnum,hnum))
v_ppph_sparse = []
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
        ele = [a, b, c, d, val]
        if ele not in v_pppp_sparse:
            v_pppp_sparse.append(ele)
    elif p in part and q in part and r in part and s in hole:
        a=part.index(p)
        b=part.index(q)
        c=part.index(r)
        d=hole.index(s)
        v_ppph[a,b,c,d]=val
        ele = [a, b, c, d, val]
        if ele not in v_ppph_sparse:
            v_ppph_sparse.append(ele)
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
print('Data done reading')

#organizing data into lists
fock_mats = [f_pp, f_ph, f_hh]
two_bod_dense = [v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh]
two_bod_sparse = [v_pppp_sparse, v_ppph_sparse, v_pphh, v_phph, v_phhh, v_hhhh]

#Run both sparse and dense calculations
print('Dense Calculation')
corrEnDense, t1Dense, t2Dense = ccm.ccsd_solver(fock_mats, two_bod_dense, verbose=True, sparse=False)
print('Sparse Calculation')
corrEnSparse, t1Sparse, t2Sparse = ccm.ccsd_solver(fock_mats, two_bod_sparse, verbose=True, sparse = True)

assert (corrEnDense - corrEnSparse) / corrEnDense < 1e-6, 'Sparse different from dense'
