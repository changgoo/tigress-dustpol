import numpy as np

class HD_dust_model(object):
    def __init__(self,datadir='./'):
        # Initialize splines
        # Do not want to call this every time S_nu is called
        self.dust_interp=self.initialize_hd_dust_model(datadir)

    def initialize_hd_dust_model(self,datadir):
        '''
        Physical dust model
        '''
        from scipy import interpolate
        # Read in precomputed dust emission spectra as a function of lambda and U
        data_sil = np.genfromtxt(datadir + "sil_fe00_2.0.dat")
        data_car = np.genfromtxt(datadir + "car_1.0.dat")
 
        c = 2.99792458e10  # Speed of light, cm/s
        wav = data_sil[:,0]
        uvec = np.arange(-3.,5.01,0.1) # log U from -3 to 5
        sil_i = interpolate.RectBivariateSpline(uvec,wav,(data_sil[:,3:84]*
                                         (wav[:,np.newaxis]*1.e-4/c)*1.e23*1.e-6).T) # to MJy/sr/NH
        car_i = interpolate.RectBivariateSpline(uvec,wav,(data_car[:,3:84]*
                                         (wav[:,np.newaxis]*1.e-4/c)*1.e23*1.e-6).T) # to MJy/sr/NH
 
        return (car_i, sil_i)


    def calculate_Snu(self,G0,nu):
        '''
        calculate S_nu from physical dust model
        '''
        import astropy.constants as const
        import astropy.units as unit
 
        (car_i, sil_i) = self.dust_interp
        uval = np.log10(G0)+0.2
        lam = (const.c/(nu*unit.GHz)).to(unit.um)
        Snu = (sil_i.ev(uval, lam) + car_i.ev(uval, lam))#*unit.MJy*unit.cm**2/unit.sr
        #*nH/unit.cm**3*deltas*const.pc
 
        return(Snu)

class simple_dust_model(object):
    def __init__(self):
        self.initialize_dust_parameters()

    def initialize_dust_parameters(self,Tdust=18):
        '''
        Dust parameters used in KCF19
        '''
        self.Tdust = Tdust # dust temperature, K
        self.sigma = 1.2e-26

        return 

    def calculate_Snu(self,G0,nu):
        '''
        calculate S_nu from simple dust model
        '''
        import astropy.constants as const
        import astropy.units as unit
        from astropy.modeling.blackbody import blackbody_nu
        Tdust = self.Tdust*unit.K 
        lam = (const.c/(nu*unit.GHz)).to(unit.um)
        Bnu = blackbody_nu(lam,Tdust)
        Snu = G0*(Bnu*self.sigma/unit.cm**2).to('MJy/(sr*cm^2)').value
        
        return(Snu)

class MBB_dust_model(object):
    def __init__(self):
        self.initialize_dust_parameters()

    def initialize_dust_parameters(self,beta=1.6,):
        '''
        Dust parameters for MBB from Hensley & Bull 2018
        '''
        self.beta = 1.6
        self.Tdust0 = 20
        self.G0 = 1.0
        self.nu0 = 353
        self.A = 3.9e-6
        self.sigma = 1.2e-26
        return 

    def calculate_Snu(self,G0,nu):
        '''
        calculate S_nu from MBB dust model
        '''
        import astropy.constants as const
        import astropy.units as unit
        from astropy.modeling.blackbody import blackbody_nu
        Tdust = self.Tdust0*(G0/self.G0)**(1/(4.+self.beta))*unit.K 
        lam = (const.c/(nu*unit.GHz)).to(unit.um)
        #lam0 = (const.c/(self.nu0*unit.GHz)).to(unit.um)
        Bnu = blackbody_nu(lam,Tdust)
        #Bnu0 = blackbody_nu(lam0,self.Tdust0)
        Snu = (nu/self.nu0)**self.beta*(Bnu*self.sigma/unit.cm**2).to('MJy/(sr*cm^2)').value
        
        return(Snu)
