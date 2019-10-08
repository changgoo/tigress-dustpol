import numpy as np
import astropy.constants as ac
import astropy.units as au
import os

class data_container(object):
    def __init__(self,mhdfile,radfile=None,dust_model='MBB'):
        from vtk_reader import AthenaDataSet
        dsmhd=AthenaDataSet(mhdfile)
        dust_model_list=['MBB','HD','simple']
        if not (dust_model in dust_model_list):
            print("{} is not a supported dust model")
            print("choose from",dust_model_list)
            return 
        self.dust_model_label = dust_model
        if radfile != None:
            self.dsrad=AthenaDataSet(radfile)
            print(dsmhd.domain['time'],self.dsrad.domain['time'])
            self.ART=True
        else:
            self.ART=False
            self.dust_model_label = 'simple'
        self.dx = dsmhd.domain['dx'][0]
        self.time = dsmhd.domain['time']
        self.dsmhd = dsmhd
        
        self._set_dust_model(dust_model)
        self._set_shear()
        
    def _set_dust_model(self,dust_model):
        from dust_models import HD_dust_model,simple_dust_model, MBB_dust_model
        if self.ART:
            if dust_model == 'HD': 
                self.dust_model = HD_dust_model('./')
                self.dust_model_label = 'HD'
            elif dust_model == 'MBB': 
                self.dust_model = MBB_dust_model()
                self.dust_model_label = 'MBB'
            else: 
                self.dustmodel = simple_dust_model()
                self.dust_model_label = 'simple'
        else:
            self.dust_model = simple_dust_model()
            self.dust_model_label = 'simple'
                
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
        dirname='./tmp'
        ds=self.dsmhd
        if not os.path.isdir(dirname): os.mkdir(dirname)
        fp = open('{}/{}.{}.nH'.format(dirname,ds.id,ds.step), 'wb')
        np.asarray([NZ, NY, NX], np.int32).tofile(fp)
        np.asarray(nH,  np.float32).tofile(fp)
        fp.close()
        np.asarray(G0,  np.float32).tofile('{}/{}.{}.G0'.format(dirname,ds.id,ds.step))
        np.asarray(Bx,  np.float32).tofile('{}/{}.{}.Bx'.format(dirname,ds.id,ds.step))
        np.asarray(By,  np.float32).tofile('{}/{}.{}.By'.format(dirname,ds.id,ds.step))
        np.asarray(Bz,  np.float32).tofile('{}/{}.{}.Bz'.format(dirname,ds.id,ds.step))

    def _load_data(self):
        dirname='./tmp'
        if not os.path.isdir(dirname): os.mkdir(dirname)
        ds=self.dsmhd
        head='{}/{}.{}'.format(dirname,ds.id,ds.step)
        NZ, NY, NX = np.fromfile('{}.nH'.format(head), np.int32, 3)
        nH  = np.fromfile('{}.nH'.format(head), np.float32)[3:].reshape(NZ, NY, NX)
        G0  = np.fromfile('{}.G0'.format(head), np.float32)
        Bx  = np.fromfile('{}.Bx'.format(head), np.float32)
        By  = np.fromfile('{}.By'.format(head), np.float32)
        Bz  = np.fromfile('{}.Bz'.format(head), np.float32)
        return nH,G0,Bx,By,Bz

    def calc_Snu(self,freq,overwrite=False):
        dirname='./Snu'
        if not os.path.isdir(dirname): os.mkdir(dirname)
        ds=self.dsmhd
        filename='{}/{}.{}.{}.{}.Snu'.format(dirname,ds.id,ds.step,freq,self.dust_model_label)
        if os.path.isfile(filename) and (not overwrite):
            Snu = np.fromfile(filename,  np.float32)
        else:
            Snu = self.dust_model.calculate_Snu(self.G0,freq)
            np.asarray(Snu,  np.float32).tofile(filename.format(freq))
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

    def write_map(self,fileinfo):
        import healpy as hp
        filename=get_mapfilename(fileinfo)
        hp.write_map(filename,[self.I,self.Q,-self.U],overwrite=True)
        print(filename)

def get_mapfilename(fileinfo):
    filename='maps/{pid}.{step}.N{nside}.f{freq}.{dust}.x{x}y{y}z{z}.IQU.fits'
    filename=filename.format(**fileinfo)
    return filename
