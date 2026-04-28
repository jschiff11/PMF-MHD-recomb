from pmhd import cons, pars

import numpy as np
from scipy.interpolate import splrep, splev, splint
from scipy.linalg import solve_banded as sb

import os

def boltfullk2(zval,karr,sourcearr,sourcenum,xe_total,steps):
    """docstring for main"""
    #--------------------------------------------------------------------------
    # Define a few background parameters
    #--------------------------------------------------------------------------

    # absorption probabilities from 2p
    pab = pars.pab(zval)
    # dimensionless doppler widths
    delta = pars.Deltah(zval)
    # voigt parameters
    a = pars.a(zval)
    # equilibrium p.s.ds
    feq = pars.feq(zval,xe_total)
    # black-body p.s.ds
    fplus = pars.bb(cons.nuly, pars.Tcmb(zval))     
    # Sobolev optical depths of the Lya line
    tauS = pars.taus(zval,xe_total)

    output_xs, dx = np.linspace( -1000.0, 1000.0, num=steps, retstep=True )

    voigt_output_xs = pars.voigt(output_xs,zval)
    
    voigt_mids = (voigt_output_xs[:-1] + voigt_output_xs[1:])/2

    #--------------------------------------------------------------------------
    # Define diagonals for banded matrix, M, s.t. M*x = b. The diagonals are
    # given in a set of matrices C, s.t. C[z][u + i - j, j] = M[z][i, j], 
    # where u is the num. of upper diagonals
    # in our case where we have derivatives which couple x_j to x_{j+1} for a given moment, u = nm
    # therefore coeffs[\cdot,nm,\cdot] \implies u + i - j = nm \implies i = j \implies diagonal entry i.e x_j
    # u + i - j = 2nm \implies i = j + nm \implies M[j+nm, j] is coupling to x_{j-1} 
    # u + i - j = 0 \implies i = j - nm \implies M[j-nm,j] is coupling to x_{j+1}
    #--------------------------------------------------------------------------

    coeffs_i = np.zeros( ( 3, len(output_xs) ) )
    # Neumann boundary condition for f0 at x-
    coeffs_i[0,1] = 1.0 
    coeffs_i[1,0] = -1.0
    # Dirichlet boundary condition at x+
    coeffs_i[1,-1] = 1.0 
    # first derivative
    coeffs_i[0,1:] = 1.0
    coeffs_i[1,:-1] = -1.0
    # second derivative
    coeffs_i[0,2:] += (tauS*(1.0 - pab)/(2.0*dx))*voigt_mids[1:]
    coeffs_i[2,:-2] = (tauS*(1.0 - pab)/(2.0*dx))*voigt_mids[:-1]
    coeffs_i[1,1:-1] += -(tauS*(1.0 - pab)/(2.0*dx))*( voigt_mids[:-1] + voigt_mids[1:] )
    
    # optical depth term 
    coeffs_i[1,1:-1] += -(tauS*pab*dx)*voigt_output_xs[1:-1]

    #--------------------------------------------------------------------------
    # Define b for M*x = b, source terms for all the f_j, at each redshift
    #--------------------------------------------------------------------------

    # source terms given as 
    # [(<delta^{(2)} x_{1s}> - <deltam^{(1)}delta^{(1)} x_{1s}>)/x_{1s}, s, <\delta f^{(2)}_{eq}> - p_{sc} <\bar{\delta f^{(2)}_{00}}> ]
    source = np.array([1.0, 1.0, 1.0])

    b_i = np.zeros( ( len(karr), len(output_xs), sourcenum ) )

    # Source term from covarriance and first moment solution
    b_i[:,1:-1] += sourcearr[:,1:-1]*dx

    moments = np.array( 
                [sb( (1,1), coeffs_i, b_i[kind] ) for kind in range( len(karr) ) ] 
                )

    # averages over line-profiles
    fphi = np.array( 
            [ 
                [splrep( output_xs, moments[kind,:,s]*voigt_output_xs ) for s in range(sourcenum)] 
                for kind in range( len(karr) ) 
                ] , dtype = object
            )
    fbar = np.array( 
            [ 
                [splint( output_xs[0], output_xs[-1], fphi[kind,s] ) for s in range(sourcenum)] 
                for kind in range( len(karr) ) 
                    ] 
                )
    

    return moments, fbar

def fullk(zval,karr,xe_total,nm,steps,psd_h):
    
    #--------------------------------------------------------------------------
    # Define a few background parameters
    #--------------------------------------------------------------------------

    # absorption probabilities from 2p
    pab = pars.pab(zval)
    # dimensionless doppler widths
    delta = pars.Deltah(zval)
    # equilibrium p.s.ds
    feq = pars.feq(zval,xe_total)
    # black-body p.s.ds
    fplus = pars.bb(cons.nuly, pars.Tcmb(zval))     
    # Sobolev optical depths of the Lya line
    tauS = pars.taus(zval,xe_total)
    
    # bin centers in terms of Doppler widths
    output_xs, dx = np.linspace( -1000.0, 1000.0, num=steps, retstep=True )
    
    voigt_output_xs = pars.voigt(output_xs,zval)
    
    voigt_mids = (voigt_output_xs[:-1] + voigt_output_xs[1:])/2
    
    # psd_h = psd
    
    #--------------------------------------------------------------------------
    # nm = 9 # number of moments followed in the Boltzmann equation, nm > 1 and odd
    #--------------------------------------------------------------------------
    # Define diagonals for banded matrix, M, s.t. M*x = b. The diagonals are
    # given in a set of matrices C, s.t. C[z][u + i - j, j] = M[z][i, j], 
    # where u is the num. of upper diagonals.
    #--------------------------------------------------------------------------

    coeffs_i = np.zeros( ( 2*nm + 1, nm*len(output_xs) ) )
    # Neumann boundary condition for f0 at x-
    coeffs_i[0,nm] = 1.0 
    coeffs_i[nm,0] = -1.0
    # Dirichlet boundary condition for all moments at x+
    coeffs_i[nm,-nm:] = 1.0 
    # first derivative for all moments
    coeffs_i[0,nm+1:] = 1.0
    coeffs_i[nm,1:-nm] = -1.0
    # second derivative for zeroth moment
    coeffs_i[0,2*nm::nm] += (tauS*(1.0 - pab)/(2.0*dx))*voigt_mids[1:]
    coeffs_i[2*nm,:-2*nm:nm] = (tauS*(1.0 - pab)/(2.0*dx))*voigt_mids[:-1]
    coeffs_i[nm,nm:-nm:nm] += -(tauS*(1.0 - pab)/(2.0*dx))*( voigt_mids[:-1] + voigt_mids[1:] )
    
    # optical depth term for j=0
    coeffs_i[nm,nm:-nm:nm] += -(tauS*pab*dx)*voigt_output_xs[1:-1]
    # optical depth term for j=1 to nm-1
    for j in range(1,nm):
        coeffs_i[nm,j:-nm:nm] += -(tauS*dx)*voigt_output_xs[:-1] 


    #--------------------------------------------------------------------------
    # Define b for M*x = b, source terms for all the f_j, at each redshift
    #--------------------------------------------------------------------------

    # source terms given as 
    # [delta n/n + delta x_{1s}/x_{1s}, \Theta/{a H}, \delta f_{eq} - p_{sc} \bar{\delta f_{00}} ]
    source = np.array([1.0, 1.0, 1.0])

    b_i = np.zeros( ( nm*len(output_xs), 3 ) )
    # source from density and ionization fraction
    b_i[nm:-nm:nm,0] = np.array( 
            [ splev( output_xs[1:-1], psd_h, der=1 )*source[0]*dx ] 
            )
    # source from velocity gradient, - sign for quadrupole goes away due to i^2
    b_i[nm:-nm:nm,1] += splev( output_xs[1:-1], psd_h, der=1 )*(-source[1]/3.0)*dx  
    b_i[2:-nm:nm,1] += splev( output_xs[:-1], psd_h, der=1 )*(2.0*source[1]/3.0)*dx 
    # source from 2p population and monopole, note sign change
    b_i[nm:-nm:nm,2] += -tauS*voigt_output_xs[1:-1]*source[2]*dx
    
    coeffs_base = coeffs_i.copy()
    moments = np.zeros( ( len(karr), nm*len(output_xs), 3 ) )
    fbar = np.zeros( (len(karr),3))
    for kind in range(len(karr)):
        coeffs_i = coeffs_base.copy()
        # list of parameters governing strength of advection, c*k*Delta/H*a
        tvsh = (
                ( cons.c*delta*(1.0 + zval) / ( pars.H(zval) ) )
                * karr[kind] ) # z by k
        # advection term for j=1 to nm-2
        for j in range(1,nm-1):
            coeffs_i[nm-1,j+1:-nm+1:nm] += (-dx*tvsh*(j+1.0)/(2.0*j+3.0))
            coeffs_i[nm+1,j-1:-nm-1:nm] += (dx*tvsh*j/(2.0*j-1.0))
        # advection term for j=0
        coeffs_i[nm-1,1+nm:-nm+1:nm] += (-(dx/3.0)*tvsh)
        # advection term for j=nm-1, with non reflecting B.C
        coeffs_i[nm,nm-1:-nm:nm] += (-dx*tvsh*(nm-1.0)/np.sqrt(4.0*(nm-1.0)**2-1.0))
        coeffs_i[nm+1,nm-2:-nm:nm] += (dx*tvsh*(nm-1.0)/(2.0*nm-3.0))

        # moments = np.array( 
        #             [sb( (nm,nm), coeffs_i, b_i )  ] 
        #             )
        moments[kind] = sb( (nm,nm), coeffs_i, b_i )  
        # averages over line-profiles
        fphi = np.array( 
                [ 
                    splrep( output_xs, moments[kind,::nm,s]*voigt_output_xs ) for s in range(3)
                    ], dtype = object
                )
        fbar[kind,:] = np.array( 
                [ 
                    splint( output_xs[0], output_xs[-1], fphi[s] ) for s in range(3)
                        ] 
                    )

    return moments, fphi, fbar

def fullkp1(zval,karr,xe_total,nm,steps,psd_h):
    #--------------------------------------------------------------------------
    # Define a few background parameters
    #--------------------------------------------------------------------------

    # dimensionless doppler widths
    delta = pars.Deltah(zval)
    # Sobolev optical depths of the Lya line
    tauS = pars.taus(zval,xe_total)
    
    # bin centers in terms of Doppler widths
    output_xs, dx = np.linspace( -1000.0, 1000.0, num=steps, retstep=True )
    
    voigt_output_xs = pars.voigt(output_xs,zval)
            

    #--------------------------------------------------------------------------
    # Define diagonals for banded matrix, M, s.t. M*x = b. The diagonals are
    # given in a set of matrices C, s.t. C[z][u + i - j, j] = M[z][i, j], 
    # where u is the num. of upper diagonals
    # in our case where we have derivatives which couple x_j to x_{j+1} for a given moment, u = nm
    # therefore coeffs[\cdot,nm,\cdot] \implies u + i - j = nm \implies i = j \implies diagonal entry i.e x_j
    # u + i - j = 2nm \implies i = j + nm \implies M[j+nm, j] is coupling to x_{j-1} 
    # u + i - j = 0 \implies i = j - nm \implies M[j-nm,j] is coupling to x_{j+1}
    #--------------------------------------------------------------------------

    coeffs_i = np.zeros( ( nm+2, nm*len(output_xs) ) )
    # Neumann boundary condition for f1 at x-
    coeffs_i[0,nm] = 1.0 
    coeffs_i[nm,0] = -1.0
    # Dirichlet boundary condition for all moments at x+
    coeffs_i[nm,-nm:] = 1.0 
    # first derivative for all moments
    coeffs_i[0,nm+1:] = 1.0
    coeffs_i[nm,1:-nm] = -1.0
    # optical depth term for j=1 to nm-1
    for j in range(1,nm):
        coeffs_i[nm,j:-nm:nm] += -(tauS*dx)*voigt_output_xs[:-1] 


    #--------------------------------------------------------------------------
    # Define b for M*x = b, source terms for all the f_j, at each redshift
    #--------------------------------------------------------------------------

    # source terms given as 
    # [delta n/n + delta x_{1s}/x_{1s}, \Theta/{a H}, \delta f_{eq} - p_{sc} \bar{\delta f_{00}} ]
    source = np.array([1.0, 1.0, 1.0])

    b_i = np.zeros( ( nm*len(output_xs), 1 ) )

    # source from rotational velocity
    b_i[2:-nm:nm,0] += splev( output_xs[:-1], psd_h, der=1 )*(-source[0]/6)*dx
    
    moments = np.zeros( ( len(karr), nm*len(output_xs), 1 ) )
    for kind in range(len(karr)):
        # list of parameters governing strength of advection, c*k*Delta/H*a
        tvsh = ( 
            ( cons.c*delta*(1.0 + zval) / ( pars.H(zval) ) )
            * karr[kind] ) # z by k
        # advection term for j=1 to nm-2
        for j in range(1,nm-1):
            coeffs_i[nm-1,j+1:-nm+1:nm] = (-dx*tvsh*(j+2)/(2.0*j+3.0))
            coeffs_i[nm+1,j-1:-nm-1:nm] = (dx*tvsh*(j-1)/(2.0*j-1.0))
        # advection term for j=nm-1, with cutoff BCs
        coeffs_i[nm+1,nm-2:-nm:nm] = (dx*tvsh*(nm-2.0)/(2.0*nm-3.0))
    
        moments[kind] = sb( (1,nm), coeffs_i, b_i )

    return moments

def fullkm1(zval,karr,xe_total,nm,steps,psd_h):
    #--------------------------------------------------------------------------
    # Define a few background parameters
    #--------------------------------------------------------------------------

    # dimensionless doppler widths
    delta = pars.Deltah(zval)
    # Sobolev optical depths of the Lya line
    tauS = pars.taus(zval,xe_total)
    
    # bin centers in terms of Doppler widths
    output_xs, dx = np.linspace( -1000.0, 1000.0, num=steps, retstep=True )
    
    voigt_output_xs = pars.voigt(output_xs,zval)
            

    #--------------------------------------------------------------------------
    # Define diagonals for banded matrix, M, s.t. M*x = b. The diagonals are
    # given in a set of matrices C, s.t. C[z][u + i - j, j] = M[z][i, j], 
    # where u is the num. of upper diagonals
    # in our case where we have derivatives which couple x_j to x_{j+1} for a given moment, u = nm
    # therefore coeffs[\cdot,nm,\cdot] \implies u + i - j = nm \implies i = j \implies diagonal entry i.e x_j
    # u + i - j = 2nm \implies i = j + nm \implies M[j+nm, j] is coupling to x_{j-1} 
    # u + i - j = 0 \implies i = j - nm \implies M[j-nm,j] is coupling to x_{j+1}
    #--------------------------------------------------------------------------

    coeffs_i = np.zeros( ( nm+2, nm*len(output_xs) ) )
    # Neumann boundary condition for f1 at x-
    coeffs_i[0,nm] = 1.0 
    coeffs_i[nm,0] = -1.0
    # Dirichlet boundary condition for all moments at x+
    coeffs_i[nm,-nm:] = 1.0 
    # first derivative for all moments
    coeffs_i[0,nm+1:] = 1.0
    coeffs_i[nm,1:-nm] = -1.0
    # optical depth term for j=1 to nm-1
    for j in range(1,nm):
        coeffs_i[nm,j:-nm:nm] += -(tauS*dx)*voigt_output_xs[:-1] 


    #--------------------------------------------------------------------------
    # Define b for M*x = b, source terms for all the f_j, at each redshift
    #--------------------------------------------------------------------------

    # source terms given as 
    # [delta n/n + delta x_{1s}/x_{1s}, \Theta/{a H}, \delta f_{eq} - p_{sc} \bar{\delta f_{00}} ]
    source = np.array([1.0, 1.0, 1.0])

    b_i = np.zeros( ( nm*len(output_xs), 1 ) )

    # source from rotational velocity
    b_i[2:-nm:nm,0] += splev( output_xs[:-1], psd_h, der=1 )*(-source[0]/6)*dx
    
    moments = np.zeros( ( len(karr), nm*len(output_xs), 1 ) )
    for kind in range(len(karr)):
        # list of parameters governing strength of advection, c*k*Delta/H*a
        tvsh = ( 
            ( cons.c*delta*(1.0 + zval) / ( pars.H(zval) ) )
            * karr[kind] ) # z by k
        # advection term for j=1 to nm-2
        for j in range(1,nm-1):
            coeffs_i[nm-1,j+1:-nm+1:nm] = (-dx*tvsh*(j)/(2.0*j+3.0))
            coeffs_i[nm+1,j-1:-nm-1:nm] = (dx*tvsh*(j+1)/(2.0*j-1.0))
        # advection term for j=nm-1, with cutoff BCs
        coeffs_i[nm+1,nm-2:-nm:nm] = (dx*tvsh*(nm)/(2.0*nm-3.0))
    
        moments[kind] = sb( (1,nm), coeffs_i, b_i )

    return moments

def boltz2fullz(z_arr,zin,zfin,sourcearr,sourcenum,xe_total,steps):
    """docstring for main"""
    #--------------------------------------------------------------------------
    # Define a few background parameters
    #--------------------------------------------------------------------------

    z_total = z_arr[zin:zfin]

    # absorption probabilities from 2p
    pab = pars.pab(z_total)
    # dimensionless doppler widths
    delta = pars.Deltah(z_total)
    # voigt parameters
    a = pars.a(z_total)
    # equilibrium p.s.ds
    feq = pars.feq(z_total,xe_total)
    # black-body p.s.ds
    fplus = pars.bb(cons.nuly, pars.Tcmb(z_total))     
    # Sobolev optical depths of the Lya line
    tauS = pars.taus(z_total,xe_total)

    output_xs, dx = np.linspace( -1000.0, 1000.0, num=steps, retstep=True )

    voigt_output_xs = pars.voigtall(output_xs,z_total)
    
    voigt_mids = (voigt_output_xs[:,:-1] + voigt_output_xs[:,1:])/2

    #--------------------------------------------------------------------------
    # Define diagonals for banded matrix, M, s.t. M*x = b. The diagonals are
    # given in a set of matrices C, s.t. C[z][u + i - j, j] = M[z][i, j], 
    # where u is the num. of upper diagonals
    # in our case where we have derivatives which couple x_j to x_{j+1} for a given moment, u = nm
    # therefore coeffs[\cdot,nm,\cdot] \implies u + i - j = nm \implies i = j \implies diagonal entry i.e x_j
    # u + i - j = 2nm \implies i = j + nm \implies M[j+nm, j] is coupling to x_{j-1} 
    # u + i - j = 0 \implies i = j - nm \implies M[j-nm,j] is coupling to x_{j+1}
    #--------------------------------------------------------------------------

    coeffs_i = np.zeros( ( len(z_total), 3, len(output_xs) ) )
    # Neumann boundary condition for f0 at x-
    coeffs_i[:,0,1] = 1.0 
    coeffs_i[:,1,0] = -1.0
    # Dirichlet boundary condition at x+
    coeffs_i[:,1,-1] = 1.0 
    # first derivative
    coeffs_i[:,0,1:] = 1.0
    coeffs_i[:,1,:-1] = -1.0
    # second derivative
    coeffs_i[:,0,2:] += (tauS*(1.0 - pab)/(2.0*dx)).reshape( (len(tauS),1) )*voigt_mids[:,1:]
    coeffs_i[:,2,:-2] = (tauS*(1.0 - pab)/(2.0*dx)).reshape( (len(tauS),1) ) *voigt_mids[:,:-1]
    coeffs_i[:,1,1:-1] += -(tauS*(1.0 - pab)/(2.0*dx)).reshape( (len(tauS),1) )*( voigt_mids[:,:-1] + voigt_mids[:,1:] )
    
    # optical depth term 
    coeffs_i[:,1,1:-1] += -(tauS*pab*dx).reshape( (len(tauS),1) ) *voigt_output_xs[:,1:-1]

    #--------------------------------------------------------------------------
    # Define b for M*x = b, source terms for all the f_j, at each redshift
    #--------------------------------------------------------------------------

    # source terms given as 
    # [(<delta^{(2)} x_{1s}> - <deltam^{(1)}delta^{(1)} x_{1s}>)/x_{1s}, s, <\delta f^{(2)}_{eq}> - p_{sc} <\bar{\delta f^{(2)}_{00}}> ]
    source = np.array([1.0, 1.0, 1.0])

    b_i = np.zeros( ( len(z_total), len(output_xs), sourcenum ) )

    # Source term from covarriance and first moment solution
    b_i[:,1:-1] += sourcearr[:,1:-1]*dx
    
    # ## Uncoment later when need to solve other sources
    # b_i = np.zeros( ( len(z_total), len(output_xs), 2 ) )
    # # source from density and ionization fraction
    # b_i[:,1:-1,0] = np.array( 
    #         [ splev( output_xs[1:-1], psd_h[z], der=1 )*dx for z in range( len(tauS) ) ] 
    #         )
    # # source from 2p population and monopole, note sign change
    # b_i[:,1:-1,1] += -tauS.reshape( (len(tauS),1) )*voigt_output_xs[:,1:-1]*dx

    moments = np.array( 
                [sb( (1,1), coeffs_i[z], b_i[z] ) for z in range( len(tauS) ) ] 
                )
    # averages over line-profiles
    fphi = np.array( 
            [ 
                [splrep( output_xs, moments[z,:,s]*voigt_output_xs[z] ) for s in range(sourcenum)] 
                for z in range( len(tauS) ) 
                ] , dtype = object
            )
    fbar = np.array( 
            [ 
                [splint( output_xs[0], output_xs[-1], fphi[z,s] ) for s in range(sourcenum)] 
                for z in range( len(tauS) ) 
                    ] 
                )

    return moments, fbar

def fullz(z_arr,zin,zfin,k,xe_total,nm,steps,psd):
    """
    Computes all moments of the scalar (m=0) perturbed boltzmann equation with Non-reflecting boundary conditions using finite difference methods. 
    Sovler goes out to \pm 1000 doppler widths from the line center.
    
    Parameters
    ----------
    z_arr : the redshift array over which you compute the moments. 
    
    For run-time efficiency purposes, it is sometimes useful to break up the redshift window. 
    zin : initial index for z_arr over which you wish to solve boltzmann equation 
    zfin : final index for z_arr over which you wish to solve boltzmann equation 
        
    k : the wavenumber for which you are solving the boltzmann equation 
    
    xe_total : the free ionizaiton fraction. It is an array that must be defined over all redshifts in z_arr.  
    
    nm-1 : the moment at which you truncate the hierarchy. Non-reflecting boundary conditions are used. 

    steps : defines the size of the doppler width array. \delta x (the doppler width in between each bin) is given by 2000/(steps-1)
    
    psd  : the homogeneous phase space density solution. For efficiency purposes, it is useful to not compute this internally and instead prescribe the psd. 
    The psd array must be defined over all redshifts in z_arr.  
    
    Returns 
    ------- 
    moments : Array of all moments for all scalar (m=0) basis solutions to the Bolzmann equaiton of shape [(zfin - zin), nm*steps, 3].
    fphi : Product of zeroth moment with voigt profile across hte line
    fbar: Array of line averaged monopoles of shape [(zfin-zin), 3] 
    
    """
    z_total = z_arr[zin:zfin]
    #--------------------------------------------------------------------------
    # Define a few background parameters
    #--------------------------------------------------------------------------

    # absorption probabilities from 2p
    pab = pars.pab(z_total)
    # dimensionless doppler widths
    delta = pars.Deltah(z_total)
    # voigt parameters
    a = pars.a(z_total)
    # equilibrium p.s.ds
    feq = pars.feq(z_total,xe_total)
    # black-body p.s.ds
    fplus = pars.bb(cons.nuly, pars.Tcmb(z_total))     
    # Sobolev optical depths of the Lya line
    tauS = pars.taus(z_total,xe_total)
    
    # bin centers in terms of Doppler widths
    output_xs, dx = np.linspace( -1000.0, 1000.0, num=steps, retstep=True )
    
    voigt_output_xs = pars.voigtall(output_xs,z_total)
    
    voigt_mids = (voigt_output_xs[:,:-1] + voigt_output_xs[:,1:])/2
    
    psd_h = psd[zin:zfin]
    
    #--------------------------------------------------------------------------

    # list of parameters governing strength of advection, c*k*Delta/H*a
    tvsh = ( 
            ( cons.c*delta*(1.0 + z_total) / ( pars.H(z_total)*cons.mpc ) )
            * k ) # z by k
    # nm = 9 # number of moments followed in the Boltzmann equation, nm > 1 and odd
    

    #--------------------------------------------------------------------------
    # Define diagonals for banded matrix, M, s.t. M*x = b. The diagonals are
    # given in a set of matrices C, s.t. C[z][u + i - j, j] = M[z][i, j], 
    # where u is the num. of upper diagonals
    # in our case where we have derivatives which couple x_j to x_{j+1} for a given moment, u = nm
    # therefore coeffs[\cdot,nm,\cdot] \implies u + i - j = nm \implies i = j \implies diagonal entry i.e x_j
    # u + i - j = 2nm \implies i = j + nm \implies M[j+nm, j] is coupling to x_{j-1} 
    # u + i - j = 0 \implies i = j - nm \implies M[j-nm,j] is coupling to x_{j+1}
    #--------------------------------------------------------------------------

    coeffs_i = np.zeros( ( len(z_total), 2*nm + 1, nm*len(output_xs) ) )
    # Neumann boundary condition for f0 at x-
    coeffs_i[:,0,nm] = 1.0 
    coeffs_i[:,nm,0] = -1.0
    # Dirichlet boundary condition for all moments at x+
    coeffs_i[:,nm,-nm:] = 1.0 
    # first derivative for all moments
    coeffs_i[:,0,nm+1:] = 1.0
    coeffs_i[:,nm,1:-nm] = -1.0
    # second derivative for zeroth moment
    coeffs_i[:,0,2*nm::nm] += (tauS*(1.0 - pab)/(2.0*dx)).reshape( (len(tauS),1) )*voigt_mids[:,1:]
    coeffs_i[:,2*nm,:-2*nm:nm] = (tauS*(1.0 - pab)/(2.0*dx)).reshape( (len(tauS),1) )*voigt_mids[:,:-1]
    coeffs_i[:,nm,nm:-nm:nm] += -(tauS*(1.0 - pab)/(2.0*dx)).reshape( (len(tauS),1) )*( voigt_mids[:,:-1] + voigt_mids[:,1:] )
    # advection term for j=0
    coeffs_i[:,nm-1,1+nm:-nm+1:nm] = (-(dx/3.0)*tvsh).reshape( (len(tauS),1) )
    # advection term for j=1 to nm-2
    for j in range(1,nm-1):
        coeffs_i[:,nm-1,j+1:-nm+1:nm] = (-dx*tvsh*(j+1.0)/(2.0*j+3.0)).reshape( (len(tauS),1) )
        coeffs_i[:,nm+1,j-1:-nm-1:nm] = (dx*tvsh*j/(2.0*j-1.0)).reshape( (len(tauS),1) )
    # advection term for j=nm-1, with non reflecting B.C
    coeffs_i[:,nm,nm-1:-nm:nm] += (-dx*tvsh*(nm-1.0)/np.sqrt(4.0*(nm-1.0)**2-1.0)).reshape( (len(tauS),1) )
    coeffs_i[:,nm+1,nm-2:-nm:nm] = (dx*tvsh*(nm-1.0)/(2.0*nm-3.0)).reshape( (len(tauS),1) )
    # optical depth term for j=0
    coeffs_i[:,nm,nm:-nm:nm] += -(tauS*pab*dx).reshape( (len(tauS),1) )*voigt_output_xs[:,1:-1]
    # optical depth term for j=1 to nm-1
    for j in range(1,nm):
        coeffs_i[:,nm,j:-nm:nm] += -(tauS*dx).reshape( (len(tauS),1) )*voigt_output_xs[:,:-1] 


    #--------------------------------------------------------------------------
    # Define b for M*x = b, source terms for all the f_j, at each redshift
    #--------------------------------------------------------------------------

    # source terms given as 
    # [delta n/n + delta x_{1s}/x_{1s}, \Theta/{a H}, \delta f_{eq} - p_{sc} \bar{\delta f_{00}} ]
    source = np.array([1.0, 1.0, 1.0])

    b_i = np.zeros( ( len(z_total), nm*len(output_xs), 3 ) )
    # source from density and ionization fraction
    b_i[:,nm:-nm:nm,0] = np.array( 
            [ splev( output_xs[1:-1], psd_h[z], der=1 )*source[0]*dx for z in range( len(tauS) ) ] 
            )
    # source from velocity gradient, - sign for quadrupole goes away due to i^2
    b_i[:,nm:-nm:nm,1] += np.array( 
            [ splev( output_xs[1:-1], psd_h[z], der=1 )*(-source[1]/3.0)*dx for z in range( len(tauS) ) ] 
            )
    b_i[:,2:-nm:nm,1] += np.array( 
            [ splev( output_xs[:-1], psd_h[z], der=1 )*(2.0*source[1]/3.0)*dx for z in range( len(tauS) ) ] 
            )
    # source from 2p population and monopole, note sign change
    b_i[:,nm:-nm:nm,2] += -tauS.reshape( (len(tauS),1) )*voigt_output_xs[:,1:-1]*source[2]*dx
    
    moments = np.array( 
                [sb( (nm,nm), coeffs_i[z], b_i[z] ) for z in range( len(tauS) ) ] 
                )
    # averages over line-profiles
    fphi = np.array( 
            [ 
                [splrep( output_xs, moments[z,::nm,s]*voigt_output_xs[z] ) for s in range(3)] 
                for z in range( len(tauS) ) 
                ], dtype = object
            )
    fbar = np.array( 
            [ 
                [splint( output_xs[0], output_xs[-1], fphi[z,s] ) for s in range(3)] 
                for z in range( len(tauS) ) 
                    ] 
                )

    return moments, fphi, fbar

def fullzp1(z_arr,zin,zfin,k,xe_total,nm,steps,psd):
    """
    Computes all moments of the m=1 perturbed boltzmann equation without nonreflecting boundary conditions (simple truncation) using finite difference methods. 
    Sovler goes out to \pm 1000 doppler widths from the line center.
    
    Parameters
    ----------
    z_arr : the redshift array over which you compute the moments. 
    
    For run-time efficiency purposes, it is sometimes useful to break up the redshift window. 
    zin : initial index for z_arr over which you wish to solve boltzmann equation 
    zfin : final index for z_arr over which you wish to solve boltzmann equation 
        
    k : the wavenumber for which you are solving the boltzmann equation 
    
    xe_total : the free ionizaiton fraction. It is an array that must be defined over all redshifts in z_arr.  
    
    nm-1 : the moment at which you truncate the hierarchy. Non-reflecting boundary conditions are used. 

    steps : defines the size of the doppler width array. \delta x (the doppler width in between each bin) is given by 2000/(steps-1)
    
    psd  : the homogeneous phase space density solution. For efficiency purposes, it is useful to not compute this internally and instead prescribe the psd. 
    The psd array must be defined over all redshifts in z_arr.  
    
    Returns 
    -------
    moments : Array of all moments for all m=1 basis solutions to the Bolzmann equaiton of shape [(zfin - zin), nm*steps, 3].
    fphi : Product of zeroth moment with voigt profile across hte line
    fbar: Array of line averaged monopoles of shape [(zfin-zin), 3] 
    
    """
    z_total = z_arr[zin:zfin]
    #--------------------------------------------------------------------------
    # Define a few background parameters
    #--------------------------------------------------------------------------

    # dimensionless doppler widths
    delta = pars.Deltah(z_total)
    # voigt parameters
    a = pars.a(z_total)
    # Sobolev optical depths of the Lya line
    tauS = pars.taus(z_total,xe_total)
    
    # bin centers in terms of Doppler widths
    output_xs, dx = np.linspace( -1000.0, 1000.0, num=steps, retstep=True )
    
    voigt_output_xs = pars.voigtall(output_xs,z_total)
    
    psd_h = psd[zin:zfin]
    
    #--------------------------------------------------------------------------

    # list of parameters governing strength of advection, c*k*Delta/H*a
    tvsh = ( 
            ( cons.c*delta*(1.0 + z_total) / ( pars.H(z_total) ) )
            * k ) # z by k
    # nm = 9 # number of moments followed in the Boltzmann equation, nm > 1 and odd
    

    #--------------------------------------------------------------------------
    # Define diagonals for banded matrix, M, s.t. M*x = b. The diagonals are
    # given in a set of matrices C, s.t. C[z][u + i - j, j] = M[z][i, j], 
    # where u is the num. of upper diagonals
    # in our case where we have derivatives which couple x_j to x_{j+1} for a given moment, u = nm
    # therefore coeffs[\cdot,nm,\cdot] \implies u + i - j = nm \implies i = j \implies diagonal entry i.e x_j
    # u + i - j = 2nm \implies i = j + nm \implies M[j+nm, j] is coupling to x_{j-1} 
    # u + i - j = 0 \implies i = j - nm \implies M[j-nm,j] is coupling to x_{j+1}
    #--------------------------------------------------------------------------

    coeffs_i = np.zeros( ( len(z_total), nm+2, nm*len(output_xs) ) )
    # Neumann boundary condition for f1 at x-
    coeffs_i[:,0,nm] = 1.0 
    coeffs_i[:,nm,0] = -1.0
    # Dirichlet boundary condition for all moments at x+
    coeffs_i[:,nm,-nm:] = 1.0 
    # first derivative for all moments
    coeffs_i[:,0,nm+1:] = 1.0
    coeffs_i[:,nm,1:-nm] = -1.0
    # advection term for j=1 to nm-2
    for j in range(1,nm-1):
        coeffs_i[:,nm-1,j+1:-nm+1:nm] = (-dx*tvsh*(j+2)/(2.0*j+3.0)).reshape( (len(tauS),1) )
        coeffs_i[:,nm+1,j-1:-nm-1:nm] = (dx*tvsh*(j-1)/(2.0*j-1.0)).reshape( (len(tauS),1) )
    # advection term for j=nm-1, with cutoff B.C
    coeffs_i[:,nm+1,nm-2:-nm:nm] = (dx*tvsh*(nm-2.0)/(2.0*nm-3.0)).reshape( (len(tauS),1) )
    # optical depth term for j=1 to nm-1
    for j in range(1,nm):
        coeffs_i[:,nm,j:-nm:nm] += -(tauS*dx).reshape( (len(tauS),1) )*voigt_output_xs[:,:-1] 


    #--------------------------------------------------------------------------
    # Define b for M*x = b, source terms for all the f_j, at each redshift
    #--------------------------------------------------------------------------

    # source terms given as 
    # [delta n/n + delta x_{1s}/x_{1s}, \Theta/{a H}, \delta f_{eq} - p_{sc} \bar{\delta f_{00}} ]
    source = np.array([1.0, 1.0, 1.0])

    b_i = np.zeros( ( len(z_total), nm*len(output_xs), 1 ) )

    # source from rotational velocity
    b_i[:,2:-nm:nm,0] += np.array( 
            [ splev( output_xs[:-1], psd_h[z], der=1 )*(-source[0]/6)*dx for z in range( len(tauS) ) ] 
            )
    
    moments = np.array( 
                [sb( (1,nm), coeffs_i[z], b_i[z] ) for z in range( len(tauS) ) ] 
                )

    return moments

def fullzm1(z_arr,zin,zfin,k,xe_total,nm,steps,psd):
    """
    Computes all moments of the m=-1 perturbed boltzmann equation without nonreflecting boundary conditions (simple truncation) using finite difference methods. 
    Sovler goes out to \pm 1000 doppler widths from the line center.
    
    Parameters
    ----------
    z_arr : the redshift array over which you compute the moments. 
    
    For run-time efficiency purposes, it is sometimes useful to break up the redshift window. 
    zin : initial index for z_arr over which you wish to solve boltzmann equation 
    zfin : final index for z_arr over which you wish to solve boltzmann equation 
        
    k : the wavenumber for which you are solving the boltzmann equation 
    
    xe_total : the free ionizaiton fraction. It is an array that must be defined over all redshifts in z_arr.  
    
    nm-1 : the moment at which you truncate the hierarchy. Non-reflecting boundary conditions are used. 

    steps : defines the size of the doppler width array. \delta x (the doppler width in between each bin) is given by 2000/(steps-1)
    
    psd  : the homogeneous phase space density solution. For efficiency purposes, it is useful to not compute this internally and instead prescribe the psd. 
    The psd array must be defined over all redshifts in z_arr.  
    
    Returns 
    -------
    moments : Array of all moments for all m=-1 basis solutions to the Bolzmann equaiton of shape [(zfin - zin), nm*steps, 3].
    fphi : Product of zeroth moment with voigt profile across hte line
    fbar: Array of line averaged monopoles of shape [(zfin-zin), 3] 
    
    """
    z_total = z_arr[zin:zfin]
    #--------------------------------------------------------------------------
    # Define a few background parameters
    #--------------------------------------------------------------------------

    # dimensionless doppler widths
    delta = pars.Deltah(z_total)
    # voigt parameters
    a = pars.a(z_total)
    # Sobolev optical depths of the Lya line
    tauS = pars.taus(z_total,xe_total)
    
    # bin centers in terms of Doppler widths
    output_xs, dx = np.linspace( -1000.0, 1000.0, num=steps, retstep=True )
    
    voigt_output_xs = pars.voigtall(output_xs,z_total)
    
    psd_h = psd[zin:zfin]
    
    #--------------------------------------------------------------------------

    # list of parameters governing strength of advection, c*k*Delta/H*a
    tvsh = ( 
            ( cons.c*delta*(1.0 + z_total) / ( pars.H(z_total) ) )
            * k ) # z by k
    # nm = 9 # number of moments followed in the Boltzmann equation, nm > 1 and odd
    

    #--------------------------------------------------------------------------
    # Define diagonals for banded matrix, M, s.t. M*x = b. The diagonals are
    # given in a set of matrices C, s.t. C[z][u + i - j, j] = M[z][i, j], 
    # where u is the num. of upper diagonals
    # in our case where we have derivatives which couple x_j to x_{j+1} for a given moment, u = nm
    # therefore coeffs[\cdot,nm,\cdot] \implies u + i - j = nm \implies i = j \implies diagonal entry i.e x_j
    # u + i - j = 2nm \implies i = j + nm \implies M[j+nm, j] is coupling to x_{j-1} 
    # u + i - j = 0 \implies i = j - nm \implies M[j-nm,j] is coupling to x_{j+1}
    #--------------------------------------------------------------------------

    coeffs_i = np.zeros( ( len(z_total), nm+2, nm*len(output_xs) ) )
    # Neumann boundary condition for f1 at x-
    coeffs_i[:,0,nm] = 1.0 
    coeffs_i[:,nm,0] = -1.0
    # Dirichlet boundary condition for all moments at x+
    coeffs_i[:,nm,-nm:] = 1.0 
    # first derivative for all moments
    coeffs_i[:,0,nm+1:] = 1.0
    coeffs_i[:,nm,1:-nm] = -1.0
    # advection term for j=1 to nm-2
    for j in range(1,nm-1):
        coeffs_i[:,nm-1,j+1:-nm+1:nm] = (-dx*tvsh*(j)/(2.0*j+3.0)).reshape( (len(tauS),1) )
        coeffs_i[:,nm+1,j-1:-nm-1:nm] = (dx*tvsh*(j+1)/(2.0*j-1.0)).reshape( (len(tauS),1) )
    # advection term for j=nm-1, with cutoff BCs
    coeffs_i[:,nm+1,nm-2:-nm:nm] = (dx*tvsh*(nm)/(2.0*nm-3.0)).reshape( (len(tauS),1) )
    # optical depth term for j=1 to nm-1
    for j in range(1,nm):
        coeffs_i[:,nm,j:-nm:nm] += -(tauS*dx).reshape( (len(tauS),1) )*voigt_output_xs[:,:-1] 


    #--------------------------------------------------------------------------
    # Define b for M*x = b, source terms for all the f_j, at each redshift
    #--------------------------------------------------------------------------

    # source terms given as 
    # [delta n/n + delta x_{1s}/x_{1s}, \Theta/{a H}, \delta f_{eq} - p_{sc} \bar{\delta f_{00}} ]
    source = np.array([1.0, 1.0, 1.0])

    b_i = np.zeros( ( len(z_total), nm*len(output_xs), 1 ) )

    # source from rotational velocity
    b_i[:,2:-nm:nm,0] += np.array( 
            [ splev( output_xs[:-1], psd_h[z], der=1 )*(-source[0]/6)*dx for z in range( len(tauS) ) ] 
            )
    
    moments = np.array( 
                [sb( (1,nm), coeffs_i[z], b_i[z] ) for z in range( len(tauS) ) ] 
                )

    return moments