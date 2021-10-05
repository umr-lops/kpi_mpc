# -*- coding: utf-8 -*-

'''
; NAME:
;   gmf_cmod5n
;
;  Translated from FORTRAN to IDL by P. Fabry 
;  the *CMOD5_N* GEOPHYSICAL MODEL
;  FUNCTION FOR THE ERS AND ASCAT C-BAND SCATTEROMETERS by
;     A. STOFFELEN              MAY  1991 ECMWF  CMOD4
;     A. STOFFELEN, S. DE HAAN  DEC  2001 KNMI   CMOD5 PROTOTYPE
;     H. HERSBACH               JUNE 2002 ECMWF  COMPLETE REVISION
;     J. de Kloe                JULI 2003 KNMI,  rewritten in fortan90
;     A. Verhoef                JAN  2008 KNMI,  CMOD5 for neutral winds
;
; PURPOSE:
;  SIMULATION OF BACKSCATTER GIVEN A WIND VECTOR AND INCIDENCE
;  ANGLE. CMOD5_N IS TO BE USED IN AN INVERSION ROUTINE IN ORDER TO
;  OBTAIN WIND VECTORS FROM BACKSCATTER TRIPLETS.
;
; INPUTS:
;  theta in [deg] incidence angle
;  v     in [m/s] wind velocity (always >= 0)
;  phi   in [deg] angle between azimuth and wind direction (= D - AZM)
;
; OUTPUTS:
;  CMOD5_N NORMALIZED BACKSCATTER (LINEAR)
;
; MODIFICATION HISTORY:
; Creation P. Fabry
; A. Mouche to Inserted the code into gmf object
; P. Vincent translated to Python

; PRINTING
;  codetopdf cmod5n.pro
;  cp /tmp/code.pdf ./code2pdf/cmod5n.pdf
;  lp ./code2pdf/cmod5n.pdf
;  evince ./code2pdf/cmod5n.pdf
;
'''
from gmf_base import GMFBase
import numpy as np


class GMFCmod5n(GMFBase):
    def __init__(self):
         super(GMFCmod5n, self).__init__('CMOD5n', 'C', 'VV')
         self._name = 'CMOD5n'
         self.vmax=50.
         self.nb_sol=2
  
    
    def returnB(self, inc_angle, wind_speed):
        """
        ; Renvoie une NRCS prédite par modèle Cmod5n
        ;
        ; INPUTS:
        ;  inc_angle:
        ;  wind_speed:
        ;  wind_dir:
        ;
        ; OPTIONAL INPUTS:
        ;  wind_speed:
        ;
        """
        theta = np.array(inc_angle)
        V     = np.array(wind_speed)
        


        THETM  = 40.
        THETHR = 25.
        

         # Coefficients obtained from Hans Hersbach by Email on 5 Feb 2008
        C = [-0.6878, -0.7957,  0.3380, -0.1728, 0.0000,  0.0040, 0.1103, 
              0.0159,  6.7329,  2.7713, -2.2885, 0.4971, -0.7250, 0.0450, 
              0.0066,  0.3222,  0.0120, 22.7000, 2.0813,  3.0000, 8.3659, 
              -3.3428,  1.3236,  6.2437,  2.3893, 0.3249,  4.1590, 1.6930]

        Y0 = C[18]
        PN = C[19]
        A  = C[18]-(C[18]-1.)/C[19]
        B  = 1./(C[19]*(C[18]-1.)**(C[19]-1.))

         #;; ANGLES
        

        X  = (theta - THETM) / THETHR
        XX = X*X

         # B0: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
        A0 =C[0]+C[1]*X+C[2]*XX+C[3]*X*XX
        A1 =C[4]+C[5]*X
        A2 =C[6]+C[7]*X
        GAM=C[8]+C[9]*X+C[10]*XX
        S0 =C[11]+C[12]*X+0*V
        S = A2*V+0*X

        w1 = np.where (S >= S0)
        w2 = np.where (S < S0)
        ss = S*0.
        ss[w1] = S[w1]
        ss[w2] = S0[w2]
        a3      = 1./(1.+ np.exp(-ss))
 
        a3[w2] = a3[w2]*(S[w2]/S0[w2])**(S0[w2]*(1-a3[w2]))


        B0=(a3**GAM)*10.**(A0+A1*V)
         # B1: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
        B1 = C[14]*V*(0.5+X-np.tanh(4.*(X+C[15]+C[16]*V)))
        B1 = C[13]*(1.+X)- B1
        B1 = B1/(np.exp(0.34*(V-C[17]))+1.)

         #B2: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
        V0 = C[20] + C[21]*X + C[22]*XX
        D1 = C[23] + C[24]*X + C[25]*XX
        D2 = C[26] + C[27]*X
        V2 = (V/V0+1.)
 
        w1 = np.where (V2 < Y0)
        V2[w1] = A + B*(V2[w1]-1.)**PN


        B2 = (-D1+D2*V2)*np.exp(-V2)

         #CMOD5_N: COMBINE THE THREE FOURIER TERMS
       

        return B0,B1,B2


    def _getNRCS(self, inc_angle, wind_speed, wind_dir):
        B0, B1, B2=self.returnB( inc_angle, wind_speed)
        phi   = np.array(wind_dir)
        FI=np.deg2rad(phi)
        CSFI = np.cos(FI)
        CS2FI= 2.00 * CSFI * CSFI - 1.00
        ZPOW   = 1.6
        sig = B0*(1.0+B1*CSFI+B2*CS2FI)**ZPOW
        return sig


class GMFCmod5h(GMFCmod5n):
    def __init__(self):
         super(GMFCmod5h, self).__init__()
         self._name = 'CMOD5h'
    
    
    
    def getNRCS(self, inc_angle, wind_speed, wind_dir):
        
        B0,B1,B2=super(GMFCmod5h, self).returnB(inc_angle, wind_speed)
        theta = np.array(inc_angle)
        V     = np.array(wind_speed)
        phi   = np.array(wind_dir)
        FI=np.deg2rad(phi)
        CSFI = np.cos(FI)
        CS2FI= 2.00 * CSFI * CSFI - 1.00
        ZPOW   = 1.6
        THETM  = 40.
        THETHR = 25.
        # B0 new is the same when V<10
        C = [-0.6878, -0.7957,  0.3380, -0.1728, 0.0000,  0.0040, 0.1103, 
              0.0159,  6.7329,  2.7713, -2.2885, 0.4971, -0.7250, 0.0450, 
              0.0066,  0.3222,  0.0120, 22.7000, 2.0813,  3.0000, 8.3659, 
              -3.3428,  1.3236,  6.2437,  2.3893, 0.3249,  4.1590, 1.6930]

     
         #;; ANGLES
        

        X  = (theta - THETM) / THETHR
        XX = X*X

         # B0: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
        A0 =C[0]+C[1]*X+C[2]*XX+C[3]*X*XX
        A1 =C[4]+C[5]*X
        A2 =C[6]+C[7]*X
        GAM=C[8]+C[9]*X+C[10]*XX
        S0 =C[11]+C[12]*X
        S = A2*10
                
        w1 = np.where (S >= S0)
        w2 = np.where (S < S0)
        ss = S*0.
        ss[w1] = S[w1]
        ss[w2] = S0[w2]
        ss2=S
        ss2[w1]=S0[w1]
        a3n      = 1./(1.+ np.exp(-ss))
        a3=a3n
       
        a3[w2] = a3n[w2]*(S[w2]/S0[w2])**(S0[w2]*(1-a3n[w2]))
        B0_10=(a3**GAM)*10.**(A0+A1*10)
        dB0=B0_10*(A1*np.log(10)+GAM*A2*(1-a3n)*S0/ss2)
       
        
        ids=np.where(V<=10)

        x=np.log10(V)
        vsat=theta*0.+62
        vsat[np.where(theta<31)]=2.*theta
       
        m=dB0/(1-np.log10(vsat))*10*np.log(10)
        #m=np.log10(dB0)/(1-np.log10(vsat))
        #c=0.5*np.log10(dB0)*(1-(1+np.log10(vsat))/(1-np.log10(vsat)))
        d=B0_10-m/2.*(1-np.log10(vsat))**2
        #d=B0_10-10**(m+1+c)/(m+1)
        B0n=m/2.*(x-np.log10(vsat))**2+d
        #B0n=1./(m+1)*V**(m+1)*10**c+d
        B0n[ids]=B0[ids] # B0 new is the same when V<10
        
        sig = B0*(B0n/B0+B1*CSFI+B2*CS2FI)**ZPOW
        
        
        return sig
