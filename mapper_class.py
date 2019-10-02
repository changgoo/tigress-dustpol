import numpy as np
import astropy.constants as ac
import astropy.units as au

class data_container(object):
    def __init__(self,mhdfile,radfile=None):
        from vtk_reader import AthenaDataSet
        dsmhd=AthenaDataSet(mhdfile)
        if radfile != None:
            self.dsrad=AthenaDataSet(radfile)
            print(dsmhd.domain['time'],self.dsrad.domain['time'])
            self.ART=True
        else:
            self.ART=False
        self.dx = dsmhd.domain['dx'][0]
        self.time = dsmhd.domain['time']
        self.dsmhd = dsmhd
        
        self._set_dust_model()
        self._set_shear()
        
    def _set_dust_model(self):
        if self.ART:
            from physical_dust import HD_dust_model
            self.dust_model = HD_dust_model('./')
        else:
            from simple_dust import simple_dust_model
            self.dust_model = simple_dust_model()
                
    def _set_shear(self):
        dsmhd=self.dsmhd
        self.shear  = True
        qshear = 1.0
        Omega  = 0.028
        vsh=qshear*Omega*dsmhd.domain['Lx'][0]
        yshear=(vsh*dsmhd.domain['time']) % dsmhd.domain['Lx'][1]
        jshear=yshear/dsmhd.domain['dx'][1]
        self.jshear = jshear
        
    def prepare_data(self,dump=False,load=False):
        dsmhd=self.dsmhd
        
        if load:
            nH,G0,Bx,By,Bz = self._load_data()
        else:
            nH = dsmhd.read_all_data('density')
            B = dsmhd.read_all_data('magnetic_field')
            Bx, By, Bz  = B[...,0], B[...,1], B[...,2]
            
            if self.ART:
                from set_units import set_units
                units = set_units(muH=1.4271)
                dsrad = self.dsrad
                E_FUV    =  dsrad.read_all_data('rad_energy_density1')
                J_FUV0   =  2.1e-4*au.erg/au.cm**2/au.s/au.sr
                J_FUV    =  E_FUV*(units['pressure']*ac.c/(4*np.pi*au.sr)).cgs
                G0  =  np.array((J_FUV/J_FUV0).cgs)
            else:
                G0 = np.ones_like(nH)
            if dump:
                self._dump_data(nH,G0,Bx,By,Bz)
                
        self.nH=nH
        self.G0=np.ravel(G0)
        self.Bx=np.ravel(Bx)
        self.By=np.ravel(By)
        self.Bz=np.ravel(Bz)

    def _dump_data(self,nH,G0,Bx,By,Bz):
        NZ, NY, NX = nH.shape   # we call the indices (z,y,x)
        fp = open('test.nH', 'wb')
        np.asarray([NZ, NY, NX], np.int32).tofile(fp)
        np.asarray(nH,  np.float32).tofile(fp)
        fp.close()
        np.asarray(G0,  np.float32).tofile('test.G0')
        np.asarray(Bx,  np.float32).tofile('test.Bx')
        np.asarray(By,  np.float32).tofile('test.By')
        np.asarray(Bz,  np.float32).tofile('test.Bz')

    def _load_data(self):
        NZ, NY, NX = np.fromfile('test.nH', np.int32, 3)
        nH  = np.fromfile('test.nH' , np.float32)[3:].reshape(NZ, NY, NX)
        G0  = np.fromfile('test.G0',  np.float32)
        Bx  = np.fromfile('test.Bx',  np.float32)
        By  = np.fromfile('test.By',  np.float32)
        Bz  = np.fromfile('test.Bz',  np.float32)
        return nH,G0,Bx,By,Bz

    def calc_Snu(self,freq,load=False):
        if load:
            Snu = np.fromfile('test.Snu.{}'.format(freq),  np.float32)
        else:
            Snu = self.dust_model.calculate_Snu(self.G0,freq)
            np.asarray(Snu,  np.float32).tofile('test.Snu.{}'.format(freq))
        self.Snu = Snu
        
class mapper(object):
    def __init__(self,data):
        self.data=data
        self.prepare_mapper()
        
    def prepare_mapper(self,GPU=1,nside=128,maxlos=3500.,minlos=160,p0=0.2,freq=353,ext=0):
        self.GPU    = GPU     # GPU=0 => use CPU, GPU=1 => use GPU
        self.nside  = nside   # Healpix size parameter, 12*nside*nside pixels
        self.ext    = ext     # Extinction  [1/pc] for unit H density
        self.maxlos = maxlos
        self.minlos = minlos
        self.p0     = p0
        self.freq   = freq
        
    def run_mapper(self,timing=False,obspos=[0.,0.,0.]):
        from HPmapper import PolHealpixMapper
        if timing: t0 = time.time()
        data = self.data
        Snu = data.Snu*ac.pc.cgs.value
        nH = data.nH
        Bx = data.Bx
        By = data.By
        Bz = data.Bz
        I, Q, U, N  = PolHealpixMapper(data.dx, self.nside, self.ext, obspos, 
                                       nH, Snu, Bx, By, Bz, GPU=self.GPU,
                                       y_shear=data.jshear, maxlos=self.maxlos, 
                                       minlos=self.minlos,
                                       p0=self.p0, polred=0)
        self.I = I
        self.Q = Q
        self.U = U
        self.N = N
        if timing: print("PolHealpixMapper: %.2f seconds" % (time.time()-t0))
