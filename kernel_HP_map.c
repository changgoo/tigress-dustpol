#define EPS   2.5e-4f
#define PEPS  5.0e-4f

#define PI        3.1415926536f
#define TWOPI     6.2831853072f
#define TWOTHIRD  0.6666666667f
#define PIHALF    1.5707963268f


int Angles2PixelRing(float phi, float theta) {
   // Convert angles to Healpix pixel index, theta=0.5*PI-latitude, phi=longitude.
   // The Healpix map must be in RING order.
   int     nl2, nl4, ncap, npix, jp, jm, ipix1 ;
   float   z, za, tt, tp, tmp ;
   int     ir, ip, kshift ;
   if ((theta<0.0f)||(theta>PI)) return -1 ;
   z   = cos(theta) ;
   za  = fabs(z) ;
   if (phi>=TWOPI)  phi -= TWOPI ;
   if (phi<0.0f)    phi += TWOPI ;
   tt  = phi / PIHALF ;    // in [0,4)
   nl2 = 2*NSIDE ;
   nl4 = 4*NSIDE ;
   ncap  = nl2*(NSIDE-1) ;   //  number of pixels in the north polar cap
   npix  = 12*NSIDE*NSIDE ;
   if (za<=TWOTHIRD) {  //  Equatorial region ------------------
      jp = (int)(NSIDE*(0.5f+tt-z*0.75f)) ;   // ! index of  ascending edge line
      jm = (int)(NSIDE*(0.5f+tt+z*0.75f)) ;   // ! index of descending edge line
      ir = NSIDE + 1 + jp - jm ;   // ! in {1,2n+1} (ring number counted from z=2/3)
      kshift = 0 ;
      if (ir%2==0) kshift = 1 ;    // ! kshift=1 if ir even, 0 otherwise
      ip = (int)( ( jp+jm - NSIDE + kshift + 1 ) / 2 ) + 1 ;   // ! in {1,4n}
      if (ip>nl4) ip -=  nl4 ;
      ipix1 = ncap + nl4*(ir-1) + ip ;
   } else {  // ! North & South polar caps -----------------------------
      tp  = tt - (int)(tt)  ;    // !MOD(tt,1.d0)
      tmp = sqrt( 3.0f*(1.0f - za) ) ;
      jp = (int)(NSIDE*tp         *tmp ) ;   // ! increasing edge line index
      jm = (int)(NSIDE*(1.0f - tp)*tmp ) ;   // ! decreasing edge line index
      ir = jp + jm + 1  ;          // ! ring number counted from the closest pole
      ip = (int)( tt * ir ) + 1 ;    // ! in {1,4*ir}
      if (ip>(4*ir)) ip -= 4*ir ;
      ipix1 = 2*ir*(ir-1) + ip ;
      if (z<=0.0f) {
         ipix1 = npix - 2*ir*(ir+1) + ip ;
      }
   }
   return  ( ipix1 - 1 ) ;    // ! in {0, npix-1} -- return the pixel index
}


bool Pixel2AnglesRing(const int ipix, float *phi, float *theta) {
   // Convert Healpix pixel index to angles (phi, theta), theta=0.5*pi-lat, phi=lon
   // Map must be in RING order.
   int    nl2, nl4, npix, ncap, iring, iphi, ip, ipix1 ;
   float  fact1, fact2, fodd, hip, fihip ;
   npix = 12*NSIDE*NSIDE;      // ! total number of points
   // if ((ipix<0)||(ipix>(npix-1))) return false ;
   ipix1 = ipix + 1 ;    // ! in {1, npix}
   nl2   = 2*NSIDE ;
   nl4   = 4*NSIDE ;
   ncap  = 2*NSIDE*(NSIDE-1) ;  // ! points in each polar cap, =0 for NSIDE =1
   fact1 = 1.5f*NSIDE ;
   fact2 = 3.0f*NSIDE*NSIDE  ;
   if (ipix1<=ncap) {   // ! North Polar cap -------------
      hip   = ipix1/2.0f ;
      fihip = (int)( hip ) ;
      iring = (int)( sqrt( hip - sqrt(fihip) ) ) + 1 ;  // ! counted from North pole
      iphi  = ipix1 - 2*iring*(iring - 1) ;
      *theta= acos( 1.0f - iring*iring / fact2 ) ;
      *phi  = (iphi - 0.5f) * PI/(2.0f*iring) ;
   } else {
      if (ipix1<=nl2*(5*NSIDE+1)) { // ! Equatorial region ------
         ip    = ipix1 - ncap - 1 ;
         iring = (int)( ip / nl4 ) + NSIDE ;   // ! counted from North pole
         iphi  = (ip%nl4) + 1 ;
         fodd  = 0.5f * (1 + (iring+NSIDE)%2) ;  // ! 1 if iring+NSIDE is odd, 1/2 otherwise
         *theta= acos( (nl2 - iring) / fact1 ) ;
         *phi  = (iphi - fodd) * PI /(2.0f*NSIDE) ;
      } else {   // ! South Polar cap -----------------------------------
         ip    = npix - ipix1 + 1 ;
         hip   = ip/2.0f ;
         fihip = (int) ( hip ) ;
         iring = (int)( sqrt( hip - sqrt(fihip) ) ) + 1 ;   // ! counted from South pole
         iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1)) ;
         *theta= acos( -1.0f + iring*iring / fact2 ) ;
         *phi  = (iphi - 0.5f) * PI/(2.0f*iring) ;
      }
   }
   return true ;
}



int Index(float3 *POS) {
   // Return 1d index for the current position in POS. Returns -1 for positions
   // outside the model volume
   int i=floor(POS->x), j=floor(POS->y), k=floor(POS->z) ;
   if ((i<0)||(j<0)||(k<0)||(i>=NX)||(j>=NY)||(k>=NZ)) return -1 ;
   return i+NX*(j+NY*k) ;   
}



float GetStep(float3 *POS, const float3 *DIR, int *ind) {
   // Calculate step to next cell, update level and ind for the next cell
   // Returns the step length in GL units (units of the root grid)
   float dx, dy, dz ;
   dx = ((((*DIR).x)>0.0f) ? ((1.0f+PEPS-fmod((*POS).x,1.0f))/((*DIR).x)) 
         : ((-PEPS-fmod((*POS).x,1.0f))/((*DIR).x))) ;
   dy = ((((*DIR).y)>0.0f) ? ((1.0f+PEPS-fmod((*POS).y,1.0f))/((*DIR).y)) 
         : ((-PEPS-fmod((*POS).y,1.0f))/((*DIR).y))) ;
   dz = ((((*DIR).z)>0.0f) ? ((1.0f+PEPS-fmod((*POS).z,1.0f))/((*DIR).z)) 
         : ((-PEPS-fmod((*POS).z,1.0f))/((*DIR).z))) ;
   dx     =  min(dx, min(dy, dz)) ;   
   *POS  +=  dx*(*DIR) ;
   *ind   =  Index(POS) ; 
   return dx ;  // step length [GL] = units where cell size == 1.0
}




__kernel void HealpixMapping(
                             const      float    EXT,    //  0 - dust extinction [1/H/GL]
                             const      float3   OPOS,   //  1 - observer position [x, y, z] as offset in [pc]
                             __global   float   *DENS,   //  2 - density cube [Nx,Ny,Nz] cells
                             __global   float   *EMIT,   //  3 - emissivity cube [Nx, Ny, Nz]
                             __global   float   *MAP,    //  4 - resulting Healpix map [12*NSIDE*NSIDE]
                             __global   float   *COLDEN  //  5 - column density map [12*NSIDE*NSIDE]
                            )
{
   const int id = get_global_id(0) ;   // one work item per map pixel
   if (id>=(12*NSIDE*NSIDE)) return ;
   float  DTAU, TAU=0.0f, PHOTONS=0.0f, NH = 0.0f, ds, theta, phi ;
   float3 POS, DIR ;
   int    ind, oind ;
   // if (id==0) printf("OPOS %8.4f %8.4f %8.4f\n", OPOS.x, OPOS.y, OPOS.z) ;
   Pixel2AnglesRing(id, &phi, &theta) ;  // one thread (work item) calculates one map pixel
   DIR.x  =  +sin(theta)*cos(phi) ;
   DIR.y  =  +sin(theta)*sin(phi) ;
   DIR.z  =   cos(theta) ;
   if (fabs(DIR.x)<1.0e-5f)      DIR.x  = -1.0e-5f ;
   if (fabs(DIR.y)<1.0e-5f)      DIR.y  = -1.0e-5f ;
   if (fabs(DIR.z)<1.0e-5f)      DIR.z  = -1.0e-5f ;
   POS.x  =   0.5*NX + OPOS.x/DX ;   // position of the observer, OPOS was offset from the centre in parsecs
   POS.y  =   0.5*NY + OPOS.y/DX ;
   POS.z  =   0.5*NZ + OPOS.z/DX ;
   if ((fmod(POS.x,1.0f)<1.0e-5f)||(fmod(POS.x,1.0f)<0.99999f)) POS.x += 2.0e-5f ;
   if ((fmod(POS.y,1.0f)<1.0e-5f)||(fmod(POS.y,1.0f)<0.99999f)) POS.y += 2.0e-5f ;
   if ((fmod(POS.z,1.0f)<1.0e-5f)||(fmod(POS.z,1.0f)<0.99999f)) POS.z += 2.0e-5f ;
   ind = Index(&POS) ;
   while (ind>=0) {   // starting with observer position, loop until the border of the model volume is reached
      oind    = ind ;                        // original cell, before step
      ds      = GetStep(&POS, &DIR, &ind) ;  // ds = length of the step, units of the grid step
      DTAU    = ds*DENS[oind]*EXT ;          // optical depth over the step, EXT = extinction / H / GL
      if (DTAU<1.0e-3f) {
         // note:  [ds]=GL, [DX]=pc, [EMIT] = emissivity / nH / pc
         PHOTONS += exp(-TAU) *  (1.0f-0.5f*DTAU)        * ds*DX*EMIT[oind]*DENS[oind] ;
      } else {
         PHOTONS += exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * ds*DX*EMIT[oind]*DENS[oind] ;
      }
      TAU    += DTAU ;            // total optical depth between current position and the observer
      NH     += ds*DENS[oind] ;   // cumulative column density
      // if (id==1) printf(" %8.4f %8.4f %8.4f\n", POS.x, POS.y, POS.z) ;
   }   
   MAP[id]    = PHOTONS ;         
   COLDEN[id] = NH*DX*3.0857e+18f ;
}




__kernel void PolHealpixMapping( 
                                const      float    EXT,     //  0 - dust extinction [1/H/GL]
                                const      float3   OPOS,    //  1 - observer position [x, y, z] as offset in [pc]
                                __global   float   *DENS,    //  2 - density cube [Nx,Ny,Nz] cells
                                __global   float   *EMIT,    //  3 - emissivity cube [Nx,Ny,Nz] cells
                                __global   float   *Bx,      //  4 - magnetic field Bx component [NX,Ny,Nz]
                                __global   float   *By,      //  5 
                                __global   float   *Bz,      //  6 
                                __global   float   *MAP,     //  8 - resulting Healpix maps [12*NSIDE*NSIDE*4]
                                const      float    Y_SHEAR  //  9 xy periodicity, shear in y direction
                               )
{
   const int id   = get_global_id(0) ;  // one work item per map pixel
   int npix = 12*NSIDE*NSIDE ;   if (id>=npix) return ;
   float DTAU, TAU=0.0f, colden = 0.0f, dens, sx, sz, phi, theta, los=0.0f ;
   float3 PHOTONS, POS, DIR ;
   int    ind, oind ;
   float3 BN  ;
   PHOTONS.x = 0.0f ;  PHOTONS.y = 0.0f ;  PHOTONS.z = 0.0f ;
   Pixel2AnglesRing(id, &phi, &theta) ;  // this work item calculates a single pixel
   DIR.x   =  +sin(theta)*cos(phi) ;
   DIR.y   =  +sin(theta)*sin(phi) ;
   DIR.z   =   cos(theta) ;
   if (fabs(DIR.x)<1.0e-5f)      DIR.x  = 1.0e-5f ;
   if (fabs(DIR.y)<1.0e-5f)      DIR.y  = 1.0e-5f ;
   if (fabs(DIR.z)<1.0e-5f)      DIR.z  = 1.0e-5f ;   
   POS.x   =   0.5*NX + OPOS.x/DX ;   // position of the observer, OPOS was offset from the centre in parsecs
   POS.y   =   0.5*NY + OPOS.y/DX ;
   POS.z   =   0.5*NZ + OPOS.z/DX ;
   if ((fmod(POS.x,1.0f)<1.0e-5f)||(fmod(POS.x,1.0f)<0.99999f)) POS.x += 2.0e-5f ;
   if ((fmod(POS.y,1.0f)<1.0e-5f)||(fmod(POS.y,1.0f)<0.99999f)) POS.y += 2.0e-5f ;
   if ((fmod(POS.z,1.0f)<1.0e-5f)||(fmod(POS.z,1.0f)<0.99999f)) POS.z += 2.0e-5f ;
   ind = Index(&POS) ;  // cell index for the observer
   float3 HRA, HDE ;    // HRA points left, HDE points to north
   HRA.x   =  -sin(phi) ;
   HRA.y   =  +cos(phi) ;
   HRA.z   =   0.0f ;
   HDE.x   =  -cos(theta)*cos(phi) ;
   HDE.y   =  -cos(theta)*sin(phi) ;
   HDE.z   =  +sin(theta) ;   
   float p =  p0 ;          // constant p0 ...  or encoded in the length of the B vector (if polred>0)
   
   while (ind>=0) {         // loop along the ray
      oind    =  ind ;      // original cell, before stepping over it
      dens    =  DENS[oind] ;
      sx      =  GetStep(&POS, &DIR, &ind) ;  // step length in GL units where one cell == 1.0 GL
      los    +=  sx  ;                        // los in grid units, to be compared with MAXLOS
      if (los>MINLOS) {
         if (los>MAXLOS) {                       // restricted integration length (MAXLOS in grid units)
            ind = -1 ;  POS.z = -1.0f ;
            sx  =  MAXLOS-(los-sx) ;             // actual step, up to distance MAXLOS, [MAXLOS]=GL
         }
         DTAU    =  sx*dens*EXT ;                // [sx]=GL, [EXT] = extinction/H/GL
         // for LOS along x,  Psi = atan2(By, Bz), gamma  = atan2(Bx, sqrt(By*By+Bz*Bz)) ;
         //   Psi    = angle east of north, in the plane of the sky --- full 2*pi !!!
         //   Psi is in IAU convention   tan(2*Psi) = U/Q, Psi = 0.5*atan2(U,Q)
         //   gamma  = angle away from the POS = 90 - angle wrt. DIR
         //    cos(gamma) = sin(complement)
         BN.x = Bx[oind] ;  BN.y = By[oind] ;  BN.z = Bz[oind] ;
#if (POLRED>0)
         p = length(BN) ; sss
#endif
         BN = normalize(BN) ;
         
         // Psi angle from North to East -- assumes that RA points to left
         // add 0.5*PI so that this becomes angle of polarised emission, not of B
         float Psi  =  0.5*PI+atan2(dot(BN, HRA), dot(BN, HDE)) ; // ANGLE FOR POLARISED EMISSION, NOT B
         float cc   =  0.99999f - 0.99998f * dot(BN, DIR)*dot(BN, DIR) ; // cos(gamma)^2
         // [sx]=GL, [EMIT]=emission/pc/H  ==>  sx*DX = step lenth in parsecs
         if (DTAU<1.0e-3f) {
            sz = exp(-TAU) *  (1.0f-0.5f*DTAU)        * (sx*DX) * EMIT[oind]*dens ;
         } else {
            sz = exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * (sx*DX) * EMIT[oind]*dens ;
         }
         
         PHOTONS.x  +=      sz * (1.0f-p*(cc-0.6666667f)) ;   // I
         PHOTONS.y  +=  p * sz * cos(2.0f*Psi)*cc ;           // Q
         PHOTONS.z  +=  p * sz * sin(2.0f*Psi)*cc ;           // U   -- IAU convention, Psi East of North
         TAU        +=  DTAU ;
         colden     +=  sx*dens ;   // GL*H, later scaled with DX => H*pc
       
         if (Y_SHEAR!=0.0f) {
            // We are dealing with a shearing box simulation -- assume periodicity in the
            // x and y directions, shear Y_SHEAR [root grid cells] between the high and the low x edges
            //  ==>  ray continues from x=0 to x=NX-1 but y-coordinate is shifted by -Y_SHEAR
            //       ray continues from x=NX-1 to x=0 but y-coordinate is shifted by +Y_SHEAR
            //  left = -Y_SHEAR, right = +Y_SHEAR
            if ((ind<0)&&(los<MAXLOS)) {           // if we reach MAXLOS, do not try to continue
               if ((POS.z>0.0f)&&(POS.z<NZ)) {     // ray exited but not on the z boundaries
                  if (POS.y<0.0f) POS.y = NY-PEPS ;
                  if (POS.y>NY)   POS.y = +PEPS ;
                  if (POS.x<0.0f) {
                     POS.x = NX-PEPS ;    POS.y = fmod(POS.y+NY-Y_SHEAR, (float)NY) ;
                  }
                  if (POS.x>NX)   {
                     POS.x = +PEPS   ;    POS.y = fmod(POS.y   +Y_SHEAR, (float)NY) ;
                  }
                  ind = Index(&POS) ;            
               }         
            }
         }
      }   
   } // while ind>=0  --- loop until ray exits the model volume or MAXLOS is reached
   
   
   // i = longitude, j = latitude = runs faster
   MAP[4*id+0] = PHOTONS.x ;                 // I
   MAP[4*id+1] = PHOTONS.y ;                 // Q
   MAP[4*id+2] = PHOTONS.z ;                 // U
   MAP[4*id+3] = colden * DX * 3.0857e+18f;  // DX == GL == cell size [pc]
   //  Q = cos(2*Psi), U = sin(2*Psi),  tan(2*Chi) = U/Q
   // we rotated Psi by 0.5*pi to make this polarisation angle, not angle of B
}
