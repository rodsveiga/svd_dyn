# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
from array import array
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, erf, acos
from libc.stdlib cimport malloc, free
import time


cdef inline double square(double x): return x*x


cdef double fs(double[:] loc_field, double[:] a):
  '''Model'''
  cdef double phi =0.
  cdef unsigned int j, L= loc_field.shape[0]
  for j in range(0,L):
    phi += a[j]*erf( loc_field[j]/ sqrt(2) ) / sqrt(L)
  return phi


cdef double ft(double[:] beta, double[:] x):
  '''Linear teacher'''
  cdef double phi =0.
  cdef unsigned int j, L= beta.shape[0]
  for j in range(0,L):
      phi += beta[j]*x[j] / sqrt(L)
  return phi


cdef double[:,:] rec(double[:,:] U, double[:] sv, double [:,:] VT):
  '''Reconstruct matrix from SVD decomposition'''
  cdef unsigned int j,l,r, d= U.shape[0], p= VT.shape[0]
  cdef double *arr = <double*>malloc(d * p * sizeof(double))
  cdef double[:,:] W = <double[:d,:p]>arr
  W[:,:] = 0.
  for l in range(0,d):
      for j in range(0,p):
          for r in range(0,p):
              W[l,j] += U[l,r]*sv[r]*VT[r,j]
  return W



cpdef cython_fit(long m_max, double[:,:] U0, double[:] sv, double[:,:] V0t, double[:] beta_t, double[:] a, double lrW, double lra, double noise, double x_scale, double tol, double t_max, plotlist, savelist, save_log, file_path_id, t_scale, save_key):

    cdef unsigned int j, q, r, r_ , u, v, j1, s, p = V0t.shape[1], d = U0.shape[0], m = 1000
    cdef double t, eg_avj, yjt, yjs, fs_test, eg_0= 0., eg_avj_old= 0., eg_, pi= acos(-1), noise_sqt = sqrt(noise), st_fac

    cdef double *arr = <double*>malloc(d * sizeof(double))
    cdef double[:] xj = <double[:d]>arr

    cdef double *arr2 = <double*>malloc(d * m * sizeof(double))
    cdef double[:,:] X_test = <double[:d,:m]>arr2

    cdef double *arr3 = <double*>malloc(p * sizeof(double))
    cdef double[:] h = <double[:p]>arr3

    cdef double *arr5 = <double*>malloc(m * sizeof(double))
    cdef double[:] ft_test = <double[:m]>arr5

    cdef list eg_av_=[], t_plot_=[]
    cdef str tol_string= False

    cdef double *arr7 = <double*>malloc(p * sizeof(double))
    cdef double[:] h0_test = <double[:p]>arr7


    cdef double *arr8 = <double*>malloc(d * p * sizeof(double))
    cdef double[:,:] W = <double[:d,:p]>arr8

    cdef double *arr12 = <double*>malloc(p * sizeof(double))
    cdef double[:] hu = <double[:p]>arr12


    '''Test set'''
    X_test = np.random.randn(d, m)
    ft_test[:] = 0.
    for r in range(0,m):
        ft_test[r] = ft(beta_t, X_test[:,r])

    t0 = time.time()
    W = rec(U0, sv, V0t)


    '''Generalization error at initialization'''
    for r in range(0,m):
        h0_test[:] = 0.
        for u in range(0,p):
            for v in range(0,d):
                h0_test[u] += W[v,u]*X_test[v][r] / sqrt(d)
        #####################################
        fs_test = fs(h0_test,a)
        eg_0 += 0.5*square(ft_test[r]-fs_test)/m


    eg_av_.append(eg_0)
    t_plot_.append(0.)
    print('d= %d,  p= %d, lrRF= %.3f, alpha= %.3f, noise= %s --- j= %d, tscale= %s, eg= %.10f, time= %.2f' % (d, p, lra, lrW/lra, '{:.0e}'.format(noise), 0, '{:.0e}'.format(0), eg_0, 0) )

    '''-------------------------------------'''

    for j in range(1,m_max+1):
        t = j/x_scale
        '''Sampling'''
        xj = np.random.randn(d)
        '''Teacher'''
        #################################
        yjt  = ft(beta_t, xj) + noise_sqt*np.random.randn()

        '''Student'''
        #####################################
        h[:] = 0.
        hu[:] = 0.
        for u in range(0,p):
            for v in range(0,d):
                h[u] += W[v,u]*xj[v] / sqrt(d)
                hu[u] += U0[v,u]*xj[v] /sqrt(d)
        #####################################
        yjs = fs(h,a)

        '''Gradient on the singular values'''
        for r in range(0,p):
            st_fac = 0.
            for j1 in range(0,p):
                #st_fac += a[j1]*np.exp(-square(h[j1])/2.)*sqrt(2./np.pi)*V0t[j1,r]
                st_fac += a[j1]*np.exp(-square(h[j1])/2.)*sqrt(2./np.pi)*V0t[r,j1]
            sv[r] += (yjt - yjs)*(lrW/p)*hu[r]*st_fac

        W = rec(U0, sv, V0t)

        '''Gradient on the a's'''
        for r in range(0,p):
            a[r] += (yjt - yjs)*(lra/p)*erf( h[r]/sqrt(2) )

        plot_j = j in plotlist
        save_j = j in savelist

        if plot_j:
            '''Generalization error'''
            eg_ = 0.
            for r in range(0,m):
                h[:] = 0.
                for u in range(0,p):
                    for v in range(0,d):
                        h[u] += W[v,u]*X_test[v][r] / sqrt(d)
                #####################################
                fs_test = fs(h,a)
                eg_ += 0.5*square(ft_test[r]-fs_test)/m

            eg_avj = eg_
            eg_av_.append(eg_avj)
            t_plot_.append(t)

            t1 = time.time()
            print('d = %d, p = %d, lrRF = %.3f, alpha = %.3f, noise= %s --- j = %d, tscale = %s, eg = %.10f, time = %.2f' % (d, p, lra, lrW/lra, '{:.0e}'.format(noise), j, '{:.0e}'.format(t), eg_avj,t1-t0) )

            t0 = t1

            if np.abs(eg_avj-eg_avj_old) < tol:
                tol_string = True
                break
            eg_avj_old = eg_avj

            if save_j and j < m_max:
                log_t_save = np.where(savelist == j)[0][0]
                if save_log:
                  id = 'LOGsave_tscale_%s_t1e%d_%s_eg_betat_Wf_af_W0_a0_t.npz' % (t_scale, log_t_save, save_key)
                  dict_save = {'eg':np.array(eg_av_), 'betat': np.array(beta_t), 'svf': np.array(sv), 'af': np.array(a), 't':np.array(t_plot_)}
                  file_path = file_path_id + '/' + id
                  np.savez(file_path, **dict_save)

    if tol_string:
        t_final = j/x_scale
        print('Terminating: | Delta(eg)| < %s (tol)' %  '{:.0e}'.format(tol) )
        id = 'FINALsave_tol%s_tscale_%s_t%s_%s_eg_betat_Wf_af_W0_a0_t.npz' % ('{:.0e}'.format(tol), t_scale, '{:.0e}'.format(t_final), save_key)
        dict_save = {'eg':np.array(eg_av_), 'betat': np.array(beta_t), 'svf': np.array(sv), 'af': np.array(a), 't':np.array(t_plot_)}
    else:
        print('Terminating: tscale = %s (t_max)' %  '{:.0e}'.format(t_max) )
        id = 'FINALsave_max_tscale_%s_t1e%d_%s_eg_betat_Wf_af_W0_a0_t.npz' % (t_scale, np.log10(t_max), save_key)
        dict_save = {'eg':np.array(eg_av_), 'betat': np.array(beta_t), 'svf': np.array(sv), 'af': np.array(a), 't':np.array(t_plot_)}

    file_path = file_path_id + '/' + id
    np.savez(file_path, **dict_save)

    return np.array(t_plot_), np.array(eg_av_), np.array(W)
