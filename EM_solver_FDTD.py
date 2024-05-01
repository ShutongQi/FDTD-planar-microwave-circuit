"""
FDTD-based planar microwave circuit modeling
Gaussian pulse source
Conductive material
Multi-resolution tune
"""

import numpy as np
import h5py

class FDTD_EM():
    
    # program constants 
    light_speed = 2.99792458e8
    mu_0 = 4.0*np.pi*1.0e-7
    eps_0 = 1.0/(light_speed**2*mu_0)
    
    def __init__(self):

        pass
    
    # Gaussian pulse as excitation
    def Gaussian_excitation(self, Ts, dt):
        t0 = 3*Ts 
        # max number of time steps for excitation
        nmax = np.round(6*Ts/dt) 
        return Ts, t0, nmax
    
    def move_grid(self, Ni, media):
        nx, ny, nz = Ni[0]+1, Ni[1]+1, Ni[2]+1
        mediaxx, mediayy, mediazz = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))

        mediaxx[0:nx,1:ny,1:nz] = 0.25*( media[0:nx,0:ny-1,1:nz]+\
                                media[0:nx,1:ny,0:nz-1]+\
                                media[0:nx,0:ny-1,0:nz-1]+\
                                media[0:nx,1:ny,1:nz] )

        mediayy[1:nx,0:ny,1:nz] = 0.25*( media[1:nx,0:ny,0:nz-1]+\
                                media[0:nx-1,0:ny,1:nz]+\
                                media[0:nx-1,0:ny,0:nz-1]+\
                                media[1:nx,0:ny,1:nz] ) 

        mediazz[1:nx,1:ny,0:nz] = 0.25*( media[1:nx,0:ny-1,0:nz]+\
                                media[0:nx-1,1:ny,0:nz]+\
                                media[0:nx-1,0:ny-1,0:nz]+\
                                media[1:nx,1:ny,0:nz] )
        
        return mediaxx, mediayy, mediazz
    
    # Adjust the index for y-axis in the PEC layer
    def tuneY(self, y):
        a, b = y.shape
        y_copy = y[:-1, :]
        afterTune = np.ones((a, b))
        for ii in range(1, a):  
            for jj in range(b):  
                if y[ii, jj] == 0 or y_copy[ii-1, jj] == 0:
                    afterTune[ii, jj] = 0
        return afterTune
    
    def maxPooling2(self, img):
        a, b = img.shape
        x = a // 2
        y = b // 2
        afterPooling = np.ones((x + 1, y + 1))
        for i in range(1, x + 1):
            for j in range(1, y + 1):
                region = img[2*i-2:2*i, 2*j-2:2*j]
                afterPooling[i-1, j-1] = np.ceil(0.25 * np.sum(region))
        return afterPooling
    
    def maxPooling3(self, img):
        a, b = img.shape
        x = a // 3
        y = b // 3
        afterPooling = np.ones((x + 1, y + 1))
        for i in range(1, x + 1):
            for j in range(1, y + 1):
                region = img[3*i-3:3*i, 3*j-3:3*j]
                afterPooling[i-1, j-1] = np.ceil(np.mean(region))
        return afterPooling
    
    # Multipliers for E-field update equations (optimized formulation)    
    def E_coefficients(self, d, Ni, epsr, sigma):
        nx, ny, nz = Ni[0]+1, Ni[1]+1, Ni[2]+1
        aex1, aey1, aez1 = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        aex2, aey2, aez2 = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        
        dx, dy, dz, dt = d[0], d[1], d[2], d[3] 
        epsxx, epsyy, epszz = self.move_grid(Ni, epsr)
        sigmaxx, sigmayy, sigmazz = self.move_grid(Ni, sigma)
        
        aex1[0:nx, 1:ny, 1:nz] = (self.eps_0*epsxx[0:nx, 1:ny, 1:nz]/dt-sigmaxx[0:nx, 1:ny, 1:nz]/2)/(self.eps_0*epsxx[0:nx, 1:ny, 1:nz]/dt+sigmaxx[0:nx, 1:ny, 1:nz]/2)
        aey1[1:nx, 0:ny, 1:nz] = (self.eps_0*epsyy[1:nx, 0:ny, 1:nz]/dt-sigmayy[1:nx, 0:ny, 1:nz]/2)/(self.eps_0*epsyy[1:nx, 0:ny, 1:nz]/dt+sigmayy[1:nx, 0:ny, 1:nz]/2)
        aez1[1:nx, 1:ny, 0:nz] = (self.eps_0*epszz[1:nx, 1:ny, 0:nz]/dt-sigmazz[1:nx, 1:ny, 0:nz]/2)/(self.eps_0*epszz[1:nx, 1:ny, 0:nz]/dt+sigmazz[1:nx, 1:ny, 0:nz]/2)

        aex2[0:nx, 1:ny, 1:nz] = dx/(dy*dz)/(self.eps_0*epsxx[0:nx, 1:ny, 1:nz]/dt+sigmaxx[0:nx, 1:ny, 1:nz]/2)
        aey2[1:nx, 0:ny, 1:nz] = dy/(dx*dz)/(self.eps_0*epsyy[1:nx, 0:ny, 1:nz]/dt+sigmayy[1:nx, 0:ny, 1:nz]/2)
        aez2[1:nx, 1:ny, 0:nz] = dz/(dx*dy)/(self.eps_0*epszz[1:nx, 1:ny, 0:nz]/dt+sigmazz[1:nx, 1:ny, 0:nz]/2)
        
        return aex1, aey1, aez1, aex2, aey2, aez2
    
    # Multipliers for H-field update equations (optimized formulation) 
    def H_coefficients(self, d):
        dx, dy, dz, dt = d[0], d[1], d[2], d[3] 
        ahx = dt*dx/(self.mu_0*dy*dz)
        ahy = dt*dy/(self.mu_0*dx*dz)
        ahz = dt*dz/(self.mu_0*dx*dy)
        
        return ahx, ahy, ahz
    
    # Multipliers for Mur's ABCs 
    def Mur_coefficients(self, epsr, d, Ni):
        nx, ny, nz = Ni[0]+1, Ni[1]+1, Ni[2]+1
        nxi, nyi, nzi = nx-1, ny-1, nz-1
        dx, dy, dz, dt = d[0], d[1], d[2], d[3] 
        epsxx, epsyy, epszz = self.move_grid(Ni, epsr)
        
        mur_coefyx, mur_coefyz, mur_coefxy, mur_coefxz = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))

        mur_coefyx[0:nxi,0,1:nzi] = (dt*self.light_speed/np.sqrt(epsxx[0:nxi,1,1:nzi])-dy)/(dt*self.light_speed/np.sqrt(epsxx[0:nxi,1,1:nzi])+dy)

        mur_coefyz[1:nxi,0,0:nzi] = (dt*self.light_speed/np.sqrt(epszz[1:nxi,1,0:nzi])-dy)/(dt*self.light_speed/np.sqrt(epszz[1:nxi,1,0:nzi])+dy)

        mur_coefxy[0,0:nyi,1:nzi] = (dt*self.light_speed/np.sqrt(epsyy[1,0:nyi,1:nzi])-dx)/(dt*self.light_speed/np.sqrt(epsyy[1,0:nyi,1:nzi])+dx)

        mur_coefxz[0,1:nyi,0:nzi] = (dt*self.light_speed/np.sqrt(epszz[1,1:nyi,0:nzi])-dx)/(dt*self.light_speed/np.sqrt(epszz[1,1:nyi,0:nzi])+dx)

        mur_coefz = (dt*self.light_speed-dz)/(dt*self.light_speed+dz)
        
        return mur_coefyx, mur_coefyz, mur_coefxy, mur_coefxz, mur_coefz
    
    # FDTD time marching
    def FDTD(self, **kwargs):
        
        # extract paramters for FDTD simulation
        d_EM, Ni_cells, Ts, steps, eps_sub, f0 = kwargs['d_EM'], kwargs['Ni_cells'], kwargs['Ts'], kwargs['steps'], kwargs['eps_sub'], kwargs['f0']
        port, inc_port = kwargs['port'], kwargs['inc_port']
        sigma, pattern = kwargs['sigma'], kwargs['PEC_layer_x']
        file_name, save_file = kwargs['file_name'], kwargs['save_file']
        tune = kwargs['tune']
        
        # dx, dy, dz, dt
        dx, dy, dz = d_EM[0]*tune, d_EM[1]*tune, d_EM[2]
        dt = 0.9/(self.light_speed)/((1/dx/dx+1/dy/dy+1/dz/dz)*2.2)**0.5
        d = np.zeros(4)
        d[0], d[1], d[2], d[3] = dx, dy, dz, dt
        
        # Nx, Ny, Nz
        nxi, nyi, nzi = int(round(Ni_cells[0]/tune)),int(round(Ni_cells[1]/tune)), int(Ni_cells[2])
        Ni= [nxi, nyi, nzi]
        nx, ny, nz = nxi+1, nyi+1, nzi+1

        # Gaussian source
        Ts, t0, nmax = self.Gaussian_excitation(Ts, dt)
        
        # excitation and sampling ports
        nyexc, nxp1, nyp1, nxp2, nyp2 = port[0], round(port[1]/tune), round(port[2]/tune), round(port[3]/tune), round(port[4]/tune)
        
        # incident ports
        x_p1, x_p2, y_p1, y_p2, z_p1 = round(inc_port[0]/tune), round(inc_port[1]/tune), round(inc_port[2]/tune), round(inc_port[3]/tune), inc_port[4]

        # Field arrays
        ex, ey, ez = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        
        hx, hy, hz = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))

        exinc, eyinc, ezinc = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))

        hxinc, hyinc, hzinc = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        
        
        # dielectric distribution
        epsr = np.zeros((nx,ny,nz))
        epsr[0:nx, 0:ny, 0:z_p1] = eps_sub
        epsr[0:nx, 0:ny, z_p1:nz]= 1.0
        
        # metallization pattern with multi-resolution adjustion
        if tune == 1:
            PEC_layer_x = pattern
        if tune == 2:
            PEC_layer_x = self.maxPooling2(pattern)
        if tune == 3:
            PEC_layer_x = self.maxPooling3(pattern)
            
        PEC_layer_y = self.tuneY(PEC_layer_x)
        
        # Multipliers for fields update and ABCs
        aex1, aey1, aez1, aex2, aey2, aez2 = self.E_coefficients(d, Ni, epsr, sigma)
        ahx, ahy, ahz = self.H_coefficients(d)
        mur_coefyx, mur_coefyz, mur_coefxy, mur_coefxz, mur_coefz = self.Mur_coefficients(epsr, d, Ni)
        
        #  Total field arrays for ABCs
        exy0, exym, exzm = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        eyx0, eyxm, eyzm = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        ezy0, ezym, ezx0, ezxm = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))

        # Incident field arrays for ABCs
        exy0inc, exyminc, exzminc = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        eyx0inc, eyxminc, eyzminc = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        ezx0inc, ezxminc, ezy0inc, ezyminc = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
        
        # sampled fields
        vinc, vtotalp1, vtotalp2 = np.zeros((steps,1)), np.zeros((steps,1)) , np.zeros((steps,1))
        
        for n in range(steps):
        #     ELECTRIC FIELD UPDATE EQUATIONS

        #     total field 
            ex[0:nxi,1:nyi,1:nzi] = aex1[0:nxi,1:nyi,1:nzi]*ex[0:nxi,1:nyi,1:nzi]+\
                                    aex2[0:nxi,1:nyi,1:nzi]*(hz[0:nxi,1:nyi,1:nzi]-hz[0:nxi,0:nyi-1,1:nzi]+\
                                                            hy[0:nxi,1:nyi,0:nzi-1]-hy[0:nxi,1:nyi,1:nzi])
            
            ey[1:nxi,0:nyi,1:nzi] = aey1[1:nxi,0:nyi,1:nzi]*ey[1:nxi,0:nyi,1:nzi]+\
                                    aey2[1:nxi,0:nyi,1:nzi]*(hx[1:nxi,0:nyi,1:nzi]-hx[1:nxi,0:nyi,0:nzi-1]+\
                                                            hz[0:nxi-1,0:nyi,1:nzi]-hz[1:nxi,0:nyi,1:nzi])
            
            ez[1:nxi,1:nyi,0:nzi] = aez1[1:nxi,1:nyi,0:nzi]*ez[1:nxi,1:nyi,0:nzi]+\
                                    aez2[1:nxi,1:nyi,0:nzi]*(hx[1:nxi,0:nyi-1,0:nzi]-hx[1:nxi,1:nyi,0:nzi]+\
                                                            hy[1:nxi,1:nyi,0:nzi]-hy[0:nxi-1,1:nyi,0:nzi])
        
        #     incident field
            exinc[0:nxi,1:nyi,1:nzi] = aex1[0:nxi,1:nyi,1:nzi]*exinc[0:nxi,1:nyi,1:nzi]+\
                                    aex2[0:nxi,1:nyi,1:nzi]*(hzinc[0:nxi,1:nyi,1:nzi]-hzinc[0:nxi,0:nyi-1,1:nzi]+\
                                                            hyinc[0:nxi,1:nyi,0:nzi-1]-hyinc[0:nxi,1:nyi,1:nzi])
            
            eyinc[1:nxi,0:nyi,1:nzi] = aey1[1:nxi,0:nyi,1:nzi]*eyinc[1:nxi,0:nyi,1:nzi]+\
                                    aey2[1:nxi,0:nyi,1:nzi]*(hxinc[1:nxi,0:nyi,1:nzi]-hxinc[1:nxi,0:nyi,0:nzi-1]+\
                                                            hzinc[0:nxi-1,0:nyi,1:nzi]-hzinc[1:nxi,0:nyi,1:nzi])
            
            ezinc[1:nxi,1:nyi,0:nzi] = aez1[1:nxi,1:nyi,0:nzi]*ezinc[1:nxi,1:nyi,0:nzi]+\
                                    aez2[1:nxi,1:nyi,0:nzi]*(hxinc[1:nxi,0:nyi-1,0:nzi]-hxinc[1:nxi,1:nyi,0:nzi]+\
                                                            hyinc[1:nxi,1:nyi,0:nzi]-hyinc[0:nxi-1,1:nyi,0:nzi])
    
        #     Perfect electric conductors - total field           
            ex[:,:,z_p1] = ex[:,:,z_p1]*PEC_layer_x
            ey[:,:,z_p1] = ey[:,:,z_p1]*PEC_layer_y

        #     Perfect electric conductors - incident field
            exinc[x_p1:x_p2-1, y_p1:y_p2, z_p1] = 0
            eyinc[x_p1:x_p2, y_p1:y_p2, z_p1] = 0
    
        #   =============================================
        #    Boundary conditions at the excitation plane 
        #   =============================================
            
            if n <= nmax:    # source is still active, apply magnetic wall
            #  total field

                ex[0:nxi,nyexc,1:nzi] = aex1[0:nxi, nyexc, 1:nzi]*ex[0:nxi, nyexc, 1:nzi] +\
                                        aex2[0:nxi, nyexc, 1:nzi]*[2*hz[0:nxi,nyexc,1:nzi]+\
                                            hy[0:nxi,nyexc,0:nzi-1]-hy[0:nxi,nyexc,1:nzi]]

                ez[1:nxi,nyexc,0:nzi] = aez1[1:nxi, nyexc,0:nzi]*ez[1:nxi,nyexc,0:nzi]+\
                                        aez2[1:nxi, nyexc,0:nzi]*[-2*hx[1:nxi,nyexc,0:nzi]+\
                                            hy[1:nxi,nyexc,0:nzi]-hy[0:nxi-1,nyexc,0:nzi]]           
                        
            # incident field
                exinc[0:nxi,nyexc,1:nzi] = aex1[0:nxi,nyexc, 1:nzi]*exinc[0:nxi, nyexc, 1:nzi] +\
                                        aex2[0:nxi, nyexc, 1:nzi]*[2*hzinc[0:nxi,nyexc,1:nzi]+\
                                            hyinc[0:nxi,nyexc,0:nzi-1]-hyinc[0:nxi,nyexc,1:nzi]]

                ezinc[1:nxi,nyexc,0:nzi] = aez1[1:nxi,nyexc,0:nzi]*ezinc[1:nxi,nyexc,0:nzi]+\
                                        aez2[1:nxi, nyexc,0:nzi]*[-2*hxinc[1:nxi,nyexc,0:nzi]+\
                                            hyinc[1:nxi,nyexc,0:nzi]-hyinc[0:nxi-1,nyexc,0:nzi]]
                
                #    Gaussian pulse application
                ez[x_p1:x_p2, nyexc, 0:z_p1] = np.exp( - ((n*dt-t0)/Ts)**2 ) 
                ezinc[x_p1:x_p2, nyexc, 0:z_p1] = np.exp( - ((n*dt-t0)/Ts)**2 ) 
    
        #     Mur's ABC for the open boundary y=0
            
        #     total field
            ex[0:nxi,0,1:nzi] = exy0[0:nxi,0,1:nzi] +\
                            mur_coefyx[0:nxi,0,1:nzi]*(ex[0:nxi,1,1:nzi]-ex[0:nxi,0,1:nzi])

            ez[1:nxi,0,0:nzi] = ezy0[1:nxi,0,0:nzi] +\
                            mur_coefyz[1:nxi,0,0:nzi]*(ez[1:nxi,1,0:nzi]-ez[1:nxi,0,0:nzi])
                        
            exy0[0:nxi,0,1:nzi] = ex[0:nxi,1,1:nzi]
            ezy0[1:nxi,0,0:nzi] = ez[1:nxi,1,0:nzi]

        #     incident field
            exinc[0:nxi,0,1:nzi] = exy0inc[0:nxi,0,1:nzi] +\
                            mur_coefyx[0:nxi,0,1:nzi]*(exinc[0:nxi,1,1:nzi]-exinc[0:nxi,0,1:nzi])

            ezinc[1:nxi,0,0:nzi] = ezy0inc[1:nxi,0,0:nzi] +\
                            mur_coefyz[1:nxi,0,0:nzi]*(ezinc[1:nxi,1,0:nzi]-ezinc[1:nxi,0,0:nzi])
                        
            exy0inc[0:nxi,0,1:nzi] = exinc[0:nxi,1,1:nzi]
            ezy0inc[1:nxi,0,0:nzi] = ezinc[1:nxi,1,0:nzi] 

    
        #     Mur's ABC for the open boundary y=ymax

        #     total field
            ex[0:nxi,nyi,1:nzi] = exym[0:nxi,0,1:nzi] +\
                            mur_coefyx[0:nxi,0,1:nzi]*(ex[0:nxi,nyi-1,1:nzi]-ex[0:nxi,nyi,1:nzi])

            ez[1:nxi,nyi,0:nzi] = ezym[1:nxi, 0, 0:nzi] +\
                            mur_coefyz[1:nxi,0,0:nzi]*(ez[1:nxi,nyi-1,0:nzi]-ez[1:nxi,nyi,0:nzi])

            exym[0:nxi,0,1:nzi] = ex[0:nxi,nyi-1,1:nzi]
            ezym[1:nxi,0,0:nzi] = ez[1:nxi,nyi-1,0:nzi]  
            
        #     incident field
            exinc[0:nxi,nyi,1:nzi] = exyminc[0:nxi,0,1:nzi] +\
                            mur_coefyx[0:nxi,0,1:nzi]*(exinc[0:nxi,nyi-1,1:nzi]-exinc[0:nxi,nyi,1:nzi])

            ezinc[1:nxi,nyi,0:nzi] = ezyminc[1:nxi, 0, 0:nzi] +\
                            mur_coefyz[1:nxi,0,0:nzi]*(ezinc[1:nxi,nyi-1,0:nzi]-ezinc[1:nxi,nyi,0:nzi])

            exyminc[0:nxi,0,1:nzi] = exinc[0:nxi,nyi-1,1:nzi]
            ezyminc[1:nxi,0,0:nzi] = ezinc[1:nxi,nyi-1,0:nzi]  
    

        #     Mur's ABC for the open boundary x=0

        #     total field     
            ey[0,0:nyi,1:nzi] = eyx0[0,0:nyi,1:nzi] +\
                            mur_coefxy[0,0:nyi,1:nzi]*(ey[1,0:nyi,1:nzi]-ey[0,0:nyi,1:nzi])

            ez[0,1:nyi,0:nzi] = ezx0[0,1:nyi,0:nzi] +\
                            mur_coefxz[0,1:nyi,0:nzi]*(ez[1,1:nyi,0:nzi]-ez[0,1:nyi,0:nzi])

            eyx0[0,0:nyi,1:nzi] = ey[1,0:nyi,1:nzi] 
            ezx0[0,1:nyi,0:nzi] = ez[1,1:nyi,0:nzi]
            
        #     incident field     
            eyinc[0,0:nyi,1:nzi] = eyx0inc[0,0:nyi,1:nzi] +\
                            mur_coefxy[0,0:nyi,1:nzi]*(eyinc[1,0:nyi,1:nzi]-eyinc[0,0:nyi,1:nzi])

            ezinc[0,1:nyi,0:nzi] = ezx0inc[0,1:nyi,0:nzi] +\
                            mur_coefxz[0,1:nyi,0:nzi]*(ezinc[1,1:nyi,0:nzi]-ezinc[0,1:nyi,0:nzi])

            eyx0inc[0,0:nyi,1:nzi] = eyinc[1,0:nyi,1:nzi] 
            ezx0inc[0,1:nyi,0:nzi] = ezinc[1,1:nyi,0:nzi]
    
        #     Mur's ABC for the open boundary x=xmax

        #     total field     
            ey[nxi,0:nyi,1:nzi] = eyxm[0,0:nyi,1:nzi] +\
                            mur_coefxy[0,0:nyi,1:nzi]*(ey[nxi-1,0:nyi,1:nzi]-ey[nxi,0:nyi,1:nzi])

            ez[nxi,1:nyi,0:nzi] = ezxm[0,1:nyi,0:nzi] +\
                            mur_coefxz[0,1:nyi,0:nzi]*(ez[nxi-1,1:nyi,0:nzi]-ez[nxi,1:nyi,0:nzi])  

            eyxm[0,0:nyi,1:nzi] = ey[nxi-1,0:nyi,1:nzi]  
            ezxm[0,1:nyi,0:nzi] = ez[nxi-1,1:nyi,0:nzi]

            
        #     incident field     
            eyinc[nxi,0:nyi,1:nzi] = eyxminc[0,0:nyi,1:nzi] +\
                            mur_coefxy[0,0:nyi,1:nzi]*(eyinc[nxi-1,0:nyi,1:nzi]-eyinc[nxi,0:nyi,1:nzi])

            ezinc[nxi,1:nyi,0:nzi] = ezxminc[0,1:nyi,0:nzi] +\
                            mur_coefxz[0,1:nyi,0:nzi]*(ezinc[nxi-1,1:nyi,0:nzi]-ezinc[nxi,1:nyi,0:nzi])  

            eyxminc[0,0:nyi,1:nzi] = eyinc[nxi-1,0:nyi,1:nzi]  
            ezxminc[0,1:nyi,0:nzi] = ezinc[nxi-1,1:nyi,0:nzi]
    
        #     Mur's ABC for the open boundary z=zmax

        #     total field     
            ex[0:nxi,1:nyi,nzi] = exzm[0:nxi,1:nyi,0] +\
                            mur_coefz*(ex[0:nxi,1:nyi,nzi-1]-ex[0:nxi,1:nyi,nzi]) 

            ey[1:nxi,0:nyi,nzi] = eyzm[1:nxi,0:nyi,0] +\
                            mur_coefz*(ey[1:nxi,0:nyi,nzi-1]-ey[1:nxi,0:nyi,nzi])
        
            exzm[0:nxi,1:nyi,0] = ex[0:nxi,1:nyi,nzi-1]
            eyzm[1:nxi,0:nyi,0] = ey[1:nxi,0:nyi,nzi-1]

            
        #     incident field     
            exinc[0:nxi,1:nyi,nzi] = exzminc[0:nxi,1:nyi,0] +\
                            mur_coefz*(exinc[0:nxi,1:nyi,nzi-1]-exinc[0:nxi,1:nyi,nzi]) 

            eyinc[1:nxi,0:nyi,nzi] = eyzminc[1:nxi,0:nyi,0] +\
                            mur_coefz*(eyinc[1:nxi,0:nyi,nzi-1]-eyinc[1:nxi,0:nyi,nzi])
        
            exzminc[0:nxi,1:nyi,0] = exinc[0:nxi,1:nyi,nzi-1]
            eyzminc[1:nxi,0:nyi,0] = eyinc[1:nxi,0:nyi,nzi-1]
    
        #     Magnetic Field Update Equations

        #     Total field
            hx[0:nxi,0:nyi,0:nzi]=hx[0:nxi,0:nyi,0:nzi]+\
                        ahx*(ey[0:nxi,0:nyi,1:nz]-ey[0:nxi,0:nyi,0:nzi]+\
                                ez[0:nxi,0:nyi,0:nzi]-ez[0:nxi,1:ny,0:nzi] )
                        
            hy[0:nxi,0:nyi,0:nzi]=hy[0:nxi,0:nyi,0:nzi]+\
                        ahy*(ex[0:nxi,0:nyi,0:nzi]-ex[0:nxi,0:nyi,1:nz]+\
                                ez[1:nx,0:nyi,0:nzi]-ez[0:nxi,0:nyi,0:nzi] )
                        
            hz[0:nxi,0:nyi,0:nzi]=hz[0:nxi,0:nyi,0:nzi]+\
                        ahz*(ex[0:nxi,1:ny,0:nzi] - ex[0:nxi,0:nyi,0:nzi]+
                                ey[0:nxi,0:nyi,0:nzi]- ey[1:nx,0:nyi,0:nzi])
            
        #     Incident field
            hxinc[0:nxi,0:nyi,0:nzi]=hxinc[0:nxi,0:nyi,0:nzi]+\
                        ahx*(eyinc[0:nxi,0:nyi,1:nz]-eyinc[0:nxi,0:nyi,0:nzi]+\
                                ezinc[0:nxi,0:nyi,0:nzi]-ezinc[0:nxi,1:ny,0:nzi] )
                        
            hyinc[0:nxi,0:nyi,0:nzi]=hyinc[0:nxi,0:nyi,0:nzi]+\
                        ahy*(exinc[0:nxi,0:nyi,0:nzi]-exinc[0:nxi,0:nyi,1:nz]+\
                                ezinc[1:nx,0:nyi,0:nzi]-ezinc[0:nxi,0:nyi,0:nzi] )
                        
            hzinc[0:nxi,0:nyi,0:nzi]=hzinc[0:nxi,0:nyi,0:nzi]+\
                        ahz*(exinc[0:nxi,1:ny,0:nzi] - exinc[0:nxi,0:nyi,0:nzi]+
                                eyinc[0:nxi,0:nyi,0:nzi]- eyinc[1:nx,0:nyi,0:nzi])
    
        #     Sampling

            vinc[n] =  ezinc[nxp1, nyp1, 2]
        
            vtotalp1[n] = ez[nxp1, nyp1, 2]
        
            vtotalp2[n] = ez[nxp2, nyp2, 2]

        #   Save the internal results every 10 steps in a h5 file
            if n%10 == 0 and save_file:
                E_output, H_output = np.array([ex,ey,ez]), np.array([hx,hy,hz])
                EM_data = np.append(E_output, H_output, axis = 0)
                data_name = 'EM_filter' + str(round(n/10))
                with h5py.File(file_name, 'a') as hdf:
                    hdf.create_dataset(data_name, data=EM_data)
        
        vtotalp1, vtotalp2, vinc =  vtotalp1.flatten(), vtotalp2.flatten(), vinc.flatten()   
        return vinc, vtotalp1, vtotalp2 
    
    def S_parameter(self, vinc, v1, v2, steps, dt):
        
        # zero padding for better frequency resolution
        N_FFT = 4*steps   
        freq = (1/dt)*np.arange(int(N_FFT/2)+1)/N_FFT 
        
        Fvinc = np.fft.fft(vinc, N_FFT) 
        Fvrefp1 = np.fft.fft(v1-vinc, N_FFT)
        Fvtotalp2 = np.fft.fft(v2, N_FFT)
        S11 = Fvrefp1/Fvinc
        S21 = Fvtotalp2/Fvinc 
        return freq, S11, S21