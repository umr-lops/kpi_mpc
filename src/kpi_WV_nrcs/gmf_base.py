# -*- coding: utf-8 -*-
import numpy as np
import math
import logging as log


def get_default_polarization_ratio_object(self, band):
    """
    TODO:
      - Grosse gestion erreur si pas de polarization etc...
      - Dans x_band_product can inverse wind  on product band,
        polarization -> true/false
      - Ajout de status ??
    """
    # bandproduct = obj_new(string(format="(%'%s_band_product')", band))
    # oPR = bandproduct->getDefaultPolarizationRatio()
    # obj_destroy, bandproduct

    #return oPR
    pass


def itertile(tile_size, startx, starty, sx, sy, factor_red=1):
    for i in range(0, (sy+tile_size)/tile_size):
        istart = starty+(i*tile_size)
        iend = istart+tile_size
        if iend > sy: iend=sy
        si=slice(istart, iend, factor_red)
        for j in range(0,(sx+tile_size)/tile_size):
            jstart = startx+(j*tile_size)
            jend = jstart + tile_size
            if jend > sx: jend=sx
            sj=slice(jstart, jend, factor_red)
            yield (si,sj)


class GMFBase(object):
    def __init__(self, name, band, polarization="VV", vmax=25.):
        self._name = name
        self._band = band
        self._polarization = polarization
        self.vmax = vmax
        self.nb_sol=1

    @property
    def name(self):
        return self._name

    @property
    def band(self):
        return self._band

    @property
    def polarization(self):
        return self._polarization


    def _getNRCS(self, inc_angle, wind_speed, wind_dir):
        """
        Methode abstraite
        Permet de définir les paramètres nécessaires pour les méthodes des
        classes dérivées.

        INPUTS:
         inc_angle:
         wind_speed:
         wind_dir:
        """
        raise NotImplementedError()

    def _getWindSpeed(self, inc_angle, sigma0, wind_dir):
        # Default getWindSpeed if no NN available
        nb_sol=self.nb_sol
        speed = np.zeros(sigma0.shape+(nb_sol,))+np.nan ###nan element (no zero value in averaging)
        dv = 10.
        v = np.arange(self.vmax*dv)/dv

        # reform sigma0 array so as to start
        # with large incidence angles
        siz = sigma0.shape
        def _iter(pos):

            sig = self._getNRCS(inc_angle[pos], v, wind_dir[pos])

            tab = sig-sigma0[pos]
            # Zero-crossing points are searched
            w = tab[:-1]*tab[1:] < 0
            
            if not w.any():
                # case 1: no zero-crossing points
                ind = np.argwhere(abs(tab)== np.min(abs(tab))).flatten()
               
                speed[pos][0:min(nb_sol,len(ind))] = v[ind[0:min(nb_sol,len(ind))]]
                
            else:
                # case 2: at least 1 zero-crossing points
                v_solut = np.zeros(w.shape)+np.nan ### nan elements (no zero value on the averaging)
                for k in np.transpose(w.nonzero()):
                    v_solut[k]= (tab[k+1]*v[k] - tab[k]*v[k+1])/ (tab[k+1]-tab[k])
                # Only 1 solution
                if np.sum(w) == 1:
                    
                    speed[pos][0] = v_solut[w]
                # More than 1 solution
                else:
                    if nb_sol==1:
                        
                        speed[pos] = np.nanmean(v_solut) # default value
                                                  # the neibourghood of the point is
                                                  # explored to select wind speed that
                                                  # is the closest
                        if len(siz) == 2: # 2D array
                            y = pos[0]
                            x = pos[1]
                            log.info("Speed Ambiguity at %d, %d, %f" % (x,y,inc_angle[pos]))
                            okernel = np.array([[-1, 0, 1]])*np.array([[1, 1, 1]]).T 
                            xx = x+okernel
                            yy = y+okernel.T
                            w2 = np.logical_and(np.logical_and(np.logical_and(xx >= 0, xx < siz[1]),
                                                               np.logical_and(yy >= 0, yy < siz[0]))
                                                ,np.logical_and(xx != x, yy != y))

                            xx = xx[w2]
                            yy = yy[w2]

                            if w2.any():
                                
                                w3 = (np.isfinite(speed[yy, xx])) ### find the not nan value in the neibourghood
                                if w3.any():
                                    tmp =  speed[xx,yy]
                                    d = np.nansum((np.outer(tmp[w3], (v_solut*0+1)) - np.outer((tmp[w3]*0+1), v_solut))**2, axis=0) ##correction  sum is on the first dimension 
                                    ind = np.nanargmin(d)
                                    speed[pos] = v_solut[ind]
                        else:
                            if len(siz) == 1: # 1D array
                                if pos > 0:
                                   
                                    xx=pos[0]-1
                                    w2 = np.logical_and(xx >= 0, xx < siz[0])
                                    if w2 :
                                        ind = np.argmin((speed[xx]-v_solut)**2.)
                                        speed[pos] = v_solut[ind]
                    else:
                        
                        speed[pos][0:max(nb_sol,np.sum(w))]=v_solut[~np.isnan(v_solut)][0:max(nb_sol,np.sum(w))]
                    

        if siz != (1,1): #bug de nditer
            it = np.nditer(sigma0, flags=['multi_index'])
            while not it.finished:
                pos = it.multi_index
                _iter(pos)
                it.iternext() ## add to pass to next position (otherwise infinite loop)
        else:
            _iter((0,0))

        speed=np.transpose(speed)
        return np.squeeze(speed)


    def getNRCS(self, inc_angle, wind_speed, wind_dir, polarization, oPolarizationRatio):
        """
        ; Methode principale de calcul de NRCS a partir des conditions
        ; d'observation.
        ; Fait appel à des méthodes spécialisées définies pour chaque classe
        ; gmf dérivant de _gmf.
        ; La conversion de NRCS en fonction de la polarisation est factorisée
        ; dans cette méthode.
        ;
        ; INPUTS:
        ;  inc_angle:
        ;  wind_speed:
        ;  wind_dir:
        ;  polarization: indication de polarization à considérer
        ;  oPolarizationRatio: reference à une instance de rapport de
        ;                      polarisation. Si l'instance n'est pas valide,
        ;                      un rapport de polarisation par défaut est
        ;                      utilisé en interne.
        ;
        """
        oPR = oPolarizationRatio
        inc = np.atleast_2d(inc_angle)
        u10 = np.atleast_2d(wind_speed)
        if u10.size == 1:
            u10 = np.ones_like(inc_angle)*wind_speed
        dir = np.atleast_2d(wind_dir)
        if dir.size == 1:
            dir = np.ones_like(inc_angle)*wind_dir

        sigma0 = self._getNRCS(inc, u10, dir)
        res = oPR.convertSigma0(sigma0, inc_angle, wind_dir,
                                fro=self._polarization,
                                to=polarization,
                                wind_speed=wind_speed)

        return res

    def getWindSpeed(self, inc_angle, sigma0, wind_dir, polarization, oPolarizationRatio):
        """
        ; Interpretation de NRCS en vent en fonction de parametres
        ; radiometriques
        ;
        ; INPUTS:
        ;  inc_angle:
        ;  sigma0:
        ;  wind_dir:
        ;  ...
        """

        if polarization is None:
            polarization = self.polarization

        if oPolarizationRatio is None:
            oPR = get_default_polarization_ratio_object(self.band)
        else:
            oPR = oPolarizationRatio

        cSigma0 = oPR.convertSigma0(sigma0, inc_angle, wind_dir,
                                    polarization, self.polarization)

        # wind retrieval may be done tile by tile when the input array is
        # much to big in size.
        tile_size = 512

        
        if len(cSigma0) == 2:
            ysize, xsize = cSigma0.shape
        else:
            ysize, xsize = cSigma0.shape[0], 1
        nb_sol=self.nb_sol
        speed = np.zeros((nb_sol,)+cSigma0.shape)
        
        if ysize <= tile_size and xsize <= tile_size:
            return np.squeeze(self._getWindSpeed(inc_angle, cSigma0, np.abs(wind_dir)))

        # traitement par tuile pour ne pas exploser la mémoire
        for sy, sx in itertile(tile_size, 0, 0, xsize, ysize):
            speed[:,sy, sx] = self._getWindSpeed(inc_angle[sy, sx],
                                               cSigma0[sy, sx],
                                               abs(wind_dir[sy, sx]))

        return np.squeeze(speed)
