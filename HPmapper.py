import sys, os
import numpy as np
import pyopencl as cl
import pyopencl.array as clarray

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def InitCL(GPU=0):
    """
    Initialise the OpenCL environment.
    Usage:
        platform, device, context, queue,  cl.mem_flags = InitCL(GPU=0)
    Input:
        GPU  =  if 1, try to get a GPU device, otherwise get a CPU device (default=0)
    Return:
        OpenCL platform, device, context, and queue and the flags object.
        If no valid devices are found, values are undefined.
    """
    platform, device, context, queue = None, None, None, None
    for iplatform in range(3):
        try:
            platform = cl.get_platforms()[iplatform]
            if (GPU>0):
                device   = platform.get_devices(cl.device_type.GPU)
            else:
                device   = platform.get_devices(cl.device_type.CPU)
            context   =  cl.Context(device)
            queue     =  cl.CommandQueue(context)
            break
        except:
            pass
    return platform, device, context, queue,  cl.mem_flags
        



def HealpixMapper(dx, nside, ext, obspos, nH, Snu, GPU=0):
    """
    Usage:
        MAP, COLDEN =  HealpixMapper(dx, nside, ext, obspos, nH, Snu, Bx, By, Bz)
    Input:
        dx      =  cell size [pc]
        nside   =  parameter of the resulting Healpix map (with 12*nside*nside pixels)
        ext     =  dust extinction [1/pc/H]
        obspos  =  position of the observer [x,y,z], relative to the centre of the model [pc]
        nH      =  density values [H], grid of [Nx, Ny, Nz] values
        Snu     =  emission/emissivity [MJy/sr/H/pc]
        GPU     =  if ==1, try to use a GPU instead of a CPU (default=0)
    Return:
        MAP     =  vector of Healpix map pixel values, for requested nside, in RING order.
        COLDEN  =  corresponding map of H column density
    """
    NZ, NY, NX  =  nH.shape
    platform, device, context, queue,  mf = InitCL(GPU)
    LOCAL       =  [8, 32][GPU>0]
    GLOBAL      =  12*nside*nside
    if (GLOBAL%LOCAL!=0):  GLOBAL = ((GLOBAL/32)+1)*32
    source      =  file("kernel_HP_map.c").read()
    # model grid dimensions (NX, NY, NZ),  Healpix map size ~ nside, model cell size ~ dx [pc]
    OPT         =  " -D NZ=%d -D NY=%d -D NX=%d -D NSIDE=%d -D DX=%.5ef -D p0=%.4ef -D MAXLOS=%.4ef" % \
                       (NZ,      NY,      NX,      nside,      dx,         0.2,     1.0e30) # p0 and MAXLOS not used
    program_map =  cl.Program(context, source).build(OPT)
    kernel_map  =  program_map.HealpixMapping
    kernel_map.set_scalar_arg_dtypes([np.float32, clarray.cltypes.float3, None, None, None, None])
    DENS_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY*NZ)
    EMIT_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY*NZ)
    MAP_buf     =  cl.Buffer(context, mf.WRITE_ONLY,  4*GLOBAL)
    COLDEN_buf  =  cl.Buffer(context, mf.WRITE_ONLY,  4*GLOBAL)
    cl.enqueue_copy(queue, DENS_buf, np.asarray(nH,  np.float32))
    cl.enqueue_copy(queue, EMIT_buf, np.asarray(Snu, np.float32))
    opos        =  clarray.vec.make_float3(obspos[0], obspos[1], obspos[2])
    extGL       =  ext*dx  # extinction per grid length, instead of per pc
    kernel_map(queue, [GLOBAL,], [LOCAL,], extGL, opos, DENS_buf, EMIT_buf, MAP_buf, COLDEN_buf)
    MAP         =  np.zeros(12*nside*nside, np.float32)
    COLDEN      =  np.zeros(12*nside*nside, np.float32)
    cl.enqueue_copy(queue, MAP,    MAP_buf)
    cl.enqueue_copy(queue, COLDEN, COLDEN_buf)
    return MAP, COLDEN
    
    
    

def PolHealpixMapper(dx, nside, ext, obspos, nH, Snu, Bx, By, Bz, GPU=0, y_shear=0.0, \
                     maxlos=1e30, p0=0.2, polred=0):
    """
    Usage:
        I, Q, U =  PolHealpixMapper(dx, nside, ext, obspos, nH, Snu, Bx, By, Bz)
    Input:
        dx      =  cell size [pc]
        nside   =  parameter of the resulting Healpix map (with 12*nside*nside pixels)
        ext     =  dust extinction [1/pc/H]
        obspos  =  position of the observer [x,y,z], relative to the centre of the model [pc]
        nH      =  density values [H], grid of [Nx, Ny, Nz] values
        Snu     =  emission/emissivity [MJy/sr/H/pc]
        Bx ...  =  magnetic field values [arbitrary units], [Nx, Ny, Nz] values each
        GPU     =  if ==1, try to use a GPU instead of a CPU (default=0)
        y_shear =  shear in y direction [cells]
        maxlos  =  maximum integration length along the LOS [pc]
        p0      =  maximum polarisation fraction, default value 0.2
        polred  =  (int) if >0, interpret |B| as polarisation fraction; default=0  => (Q,U) calculated for p=100%
    Return:
        I, Q, U, NH =  vectors of Healpix pixel values, for the requested nside, in RING order.
    Note:
        If y_shear==0.0, integration extends to the distance maxlos or to the model boundary, 
        whichever is smaller. If y_shear!=0, integration does not stop at X and Y boundaries but only
        when either MAXLOS or +/- Z boundary is reached.
    """
    NZ, NY, NX  =  nH.shape
    NPIX        =  12*nside*nside
    platform, device, context, queue,  mf = InitCL(GPU)
    LOCAL       =  [8, 32][GPU>0]
    GLOBAL      =  NPIX
    if (GLOBAL%LOCAL!=0):  GLOBAL = ((GLOBAL/32)+1)*32
    source      =  file("kernel_HP_map.c").read()
    OPT         =  \
    " -D NZ=%d -D NY=%d -D NX=%d -D NSIDE=%d -D DX=%.5ef -D MAXLOS=%.4ef -D POLRED=%d -D p0=%.4ef" % \
    (NZ, NY, NX, nside, dx, maxlos/dx, polred, p0)  # note -- in kernel [maxlos]=GL, not pc
    program_map =  cl.Program(context, source).build(OPT)
    kernel_map  =  program_map.PolHealpixMapping
    kernel_map.set_scalar_arg_dtypes([np.float32, clarray.cltypes.float3, None, None, None, None, None, None, np.float32])
    DENS_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY*NZ)
    EMIT_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY*NZ)
    Bx_buf      =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY*NZ)
    By_buf      =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY*NZ)
    Bz_buf      =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY*NZ)
    MAP_buf     =  cl.Buffer(context, mf.WRITE_ONLY,  4*4*NPIX)     # space for (I, Q, U, NH)
    #
    cl.enqueue_copy(queue, DENS_buf, np.asarray(nH,  np.float32))
    cl.enqueue_copy(queue, EMIT_buf, np.asarray(Snu, np.float32))
    cl.enqueue_copy(queue, Bx_buf,   np.asarray(Bx,  np.float32))
    cl.enqueue_copy(queue, By_buf,   np.asarray(By,  np.float32))
    cl.enqueue_copy(queue, Bz_buf,   np.asarray(Bz,  np.float32))
    opos        =  clarray.vec.make_float3(obspos[0], obspos[1], obspos[2])
    extGL       =  ext*dx  # extinction per grid unit instead of per pc
    kernel_map(queue, [GLOBAL,], [LOCAL,], extGL, opos, DENS_buf, EMIT_buf, Bx_buf, By_buf, Bz_buf, MAP_buf, y_shear)
    MAP         =  np.zeros(4*NPIX, np.float32)
    cl.enqueue_copy(queue, MAP,    MAP_buf)
    MAP.shape   = (NPIX, 4)
    return MAP[:,0], MAP[:,1], MAP[:,2], MAP[:,3] # return I, Q, U, NH

                 
    
