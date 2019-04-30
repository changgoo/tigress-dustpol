import sys, os, time

# ----------------------------------------------------------
#sys.path.append("%s/tigress-dustpol-master/" % os.getcwd())
sys.path.append(os.getcwd())
# ----------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from vtk_reader import *
import set_units
from physical_dust import HD_dust_model

units      = set_units.set_units(muH=1.4271)
dust_model = HD_dust_model('./')
ds         = AthenaDataSet('./R8_8pc_rst.rad.0300.vtk')

if (0):
    # This reads the data... and dumps to plain binary files for faster access
    dsmhd      = AthenaDataSet('./R8_8pc_rst.0300.vtk')
    for k,v in ds.domain.iteritems():
        if k != 'field_map': print(k,v)
    # read hydrogen number density and radiation energy density from './R8_8pc_rst.rad.0300.vtk' file
    # note that rad_energy_density0 is EUV field
    nH       =  ds.read_all_data('density')     #   (986, 128, 128)
    print("nH ", nH.shape)
    E_FUV    =  ds.read_all_data('rad_energy_density1')
    # converting radiation energy density to G0'
    J_FUV0   =  2.1e-4*au.erg/au.cm**2/au.s/au.sr
    J_FUV    =  E_FUV*(units['pressure']*ac.c/(4*np.pi*au.sr)).cgs
    G0prime  =  np.array((J_FUV/J_FUV0).cgs)
    # calculate S_nu at 353 GHz using Brandon's dust model
    freq     =  353
    Snu      =  dust_model.calculate_Snu(nH,G0prime,freq,deltas=ds.domain['dx'][0])
    print("Snu ", nH.shape)
    # read in magnetic fields from './R8_8pc_rst.0300.vtk' file
    B        =  dsmhd.read_all_data('magnetic_field')
    Bx, By, Bz  = B[...,0].copy(), B[...,1].copy(), B[...,2].copy()  # must be contiguous in memory...
    le       =  ds.domain['left_edge']
    re       =  ds.domain['right_edge']
    if (0):
        fig,axes =  plt.subplots(1,3,figsize=(15,5))
        extent   =  [le[0],re[0],le[1],re[1]]
        axes[0].imshow(nH.sum(axis=0),norm=LogNorm(),extent=extent)
        axes[1].imshow(G0prime.sum(axis=0),norm=LogNorm(),extent=extent)
        axes[2].imshow(Snu.value.sum(axis=0),norm=LogNorm(),extent=extent)
        plt.show()
    # dump RT data to plain binary files
    NZ, NY, NX = nH.shape   # we call the indices (z,y,x)
    fp = file('test.nH', 'wb')
    np.asarray([NZ, NY, NX], np.int32).tofile(fp)
    np.asarray(nH,  np.float32).tofile(fp)
    fp.close()
    np.asarray(Snu, np.float32).tofile('test.Snu')
    np.asarray(Bx,  np.float32).tofile('test.Bx')
    np.asarray(By,  np.float32).tofile('test.By')
    np.asarray(Bz,  np.float32).tofile('test.Bz')
    # sys.exit()
else:
    # This loads previously saved nH, Snu, Bx, By, Bz
    NZ, NY, NX = np.fromfile('test.nH', np.int32, 3)
    nH  = np.fromfile('test.nH' , np.float32)[3:].reshape(NZ, NY, NX)
    Snu = np.fromfile('test.Snu', np.float32).reshape(NZ, NY, NX)
    Bx  = np.fromfile('test.Bx',  np.float32).reshape(NZ, NY, NX)
    By  = np.fromfile('test.By',  np.float32).reshape(NZ, NY, NX)
    Bz  = np.fromfile('test.Bz',  np.float32).reshape(NZ, NY, NX)
    
        
if (0):
    plt.clf()
    plt.imshow(nH[:, :, NZ/2])
    plt.colorbar()
    plt.show()
    sys.exit()
    
    
# ========================================================================================
GPU    =  1                   # GPU=0 => use CPU, GPU=1 => use GPU
dx     =  ds.domain['dx'][0]  # Cell size [pc]
dy     =  ds.domain['dx'][1]  # Cell size [pc]
nside  =  eval(sys.argv[1])         # Healpix size parameter, 12*nside*nside pixels
ext    =   0.0                # Extinction  [1/pc] for unit H density
obspos =  [1.0, 1.0, 1.0]     # Position of the observer (x,y,z) [pc] relative to the centre of the cube
maxlos = 3500.
shear  = True
qshear = 1.0
Omega  = 0.028
tsim   = ds.domain['time']
Lx     = ds.domain['right_edge'][0] - ds.domain['left_edge'][0]
Ly     = ds.domain['right_edge'][1] - ds.domain['left_edge'][1]
qomL   = qshear*Omega*Lx
if shear:
    yshear = np.mod(qomL*tsim,Ly)/dy
else:
    yshear = 0.0
print(yshear)

from HPmapper import *
import healpy

if (0): # Plot intensity and column density
    t0 = time.time()
    M, NH = HealpixMapper(dx, nside, ext, obspos, nH, Snu, GPU=GPU)
    print("HealpixMapper: %.2f seconds" % (time.time()-t0))
    plt.figure(1, figsize=(7,6))
    healpy.mollview(M,  norm='log', min=0.001, max=30.0, fig=1, sub=221, 
                    title='Intensity', margins=[0.01, 0.01, 0.01, 0.02])
    healpy.mollview(NH, norm='log', fig=1, sub=222, 
                    title='Column density', margins=[0.01, 0.01, 0.01, 0.02])
    plt.savefig('IN.png')
else:   # Plot (I, Q, U, N)
    t0 = time.time()
    # =====================================================================
    # Convert Snu from MJy/sr/cell to  MJy/sr/H/pc as assumed by the kernel
    Snu   = np.ravel(Snu)
    Snu  /= np.ravel(nH)*dx
    # =====================================================================
    I, Q, U, N  = PolHealpixMapper(dx, nside, ext, obspos, nH, Snu, Bx, By, Bz, GPU=GPU,
                                   y_shear=yshear, maxlos=maxlos, p0=0.2, polred=0)
    print("PolHealpixMapper: %.2f seconds" % (time.time()-t0))
    ##  NSIDE=1024   CPU= 5.7 seconds, GPU=1.2 seconds
    ##  NSIDE=2048   CPU=20.1 seconds, GPU=2.2 seconds  ... 1.74 seconds with cached kernel
    plt.figure(1, figsize=(7,4.5))
    healpy.mollview(I, fig=1, sub=221, title='I', margins=[0.015, 0.020, 0.015, 0.02], norm='log', min=0.001, max=30.0)
    healpy.mollview(Q, fig=1, sub=222, title='Q', margins=[0.015, 0.020, 0.015, 0.02], norm='hist', min=-1, max=1,cmap=plt.cm.coolwarm)
    healpy.mollview(-U, fig=1, sub=223, title='U', margins=[0.015, 0.020, 0.015, 0.02], norm='hist', min=-1, max=1,cmap=plt.cm.coolwarm)
    healpy.mollview(N, fig=1, sub=224, title='N', margins=[0.015, 0.020, 0.015, 0.02], norm='log')
    plt.savefig('IQUN.png')
    plt.clf()

# =====================================================================
# With old_mapmaker

if (1):
    Snu = np.fromfile('test.Snu', np.float32).reshape(NZ, NY, NX)

    from old_mapmaker import los_dump_from_data,calc_IQU

    domain=ds.domain
    domain['shear']=True
    domain['qshear']=1.0
    domain['Omega']=0.028
    domain['fields']=['density','magnetic_field1','magnetic_field2','magnetic_field3']

    Nside=nside
    center=obspos
    smax=maxlos
    rewrite=False

    t0 = time.time()
    for data,field in zip([nH,Bx,By,Bz],['density','magnetic_field1','magnetic_field2','magnetic_field3']):
        los_dump_from_data(data,domain,dx,smax,field,Nside=Nside,center=center,force_write=rewrite)
    los_dump_from_data(Snu,domain,dx,smax,'Snu',Nside=Nside,center=center,force_write=rewrite)
    print("Remapping Data Cube: %.2f seconds" % (time.time()-t0))

    IQUN=calc_IQU(domain,dx,smax,Nside=Nside,center=center)
    print(IQUN[0].shape)

    I_cgk=IQUN[0].sum(axis=1)
    Q_cgk=IQUN[1].sum(axis=1)
    U_cgk=IQUN[2].sum(axis=1)
    N_cgk=IQUN[3].sum(axis=1)

    plt.figure(1, figsize=(7,4.5))
    healpy.mollview(I_cgk, fig=1, sub=221, title='I', margins=[0.015, 0.020, 0.015, 0.02], norm='log', min=0.001, max=30.0)
    healpy.mollview(Q_cgk, fig=1, sub=222, title='Q', margins=[0.015, 0.020, 0.015, 0.02], norm='hist', min=-1, max=1,cmap=plt.cm.coolwarm)
    healpy.mollview(U_cgk, fig=1, sub=223, title='U', margins=[0.015, 0.020, 0.015, 0.02], norm='hist', min=-1, max=1,cmap=plt.cm.coolwarm)
    healpy.mollview(N_cgk, fig=1, sub=224, title='N', margins=[0.015, 0.020, 0.015, 0.02], norm='log')
    plt.savefig('IQUN_cgk.png')
