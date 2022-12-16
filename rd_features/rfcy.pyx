# cython: boundscheck = False
# cython: cdivision = True
# cython: wraparound = False
import cython
cimport cython
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


cpdef cython_fit(long m_max, double[:,:] W0, double[:] beta_t, double[:] a, double lra, double noise, double x_scale, double tol, double alpha_max, plotlist, savelist, save_log, file_path_id, alpha_scale, save_key):
    cdef unsigned int j, q, r, p = W0.shape[1], d = W0.shape[0], m = 1000
    cdef double alpha, eg_avj, yjt, yjs, fs_test, eg_0= 0., eg_avj_old= 0., eg_, pi= acos(-1), noise_sqt = sqrt(noise)

    cdef double *arr = <double*>malloc(d * sizeof(double))
    cdef double[:] xj = <double[:d]>arr

    cdef double *arr2 = <double*>malloc(d * m * sizeof(double))
    cdef double[:,:] X_test = <double[:d,:m]>arr2

    cdef double *arr3 = <double*>malloc(p * sizeof(double))
    cdef double[:] h = <double[:p]>arr3

    cdef double *arr4 = <double*>malloc(m * sizeof(double))
    cdef double[:] ft_test = <double[:m]>arr4

    cdef double *arr5 = <double*>malloc(p * sizeof(double))
    cdef double[:] h0_test = <double[:p]>arr5

    cdef list eg_av_=[], alpha_plot_=[]
    cdef str tol_string= False

    a0 = np.copy(a)
    t0 = time.time()

    '''Test set'''
    X_test = np.random.randn(d, m)
    ft_test[:] = 0.
    for r in range(0,m):
        ft_test[r] = ft(beta_t, X_test[:,r])


    '''Generalization error at initialization'''
    for r in range(0,m):
        #h[:] = 0.
        h0_test[:] = 0.
        for u in range(0,p):
            for v in range(0,d):
                h0_test[u] += W0[v,u]*X_test[v][r] / sqrt(d)
        #####################################
        fs_test = fs(h0_test,a)
        eg_0 += 0.5*square(ft_test[r]-fs_test)/m

    eg_av_.append(eg_0)
    alpha_plot_.append(0.)
    print('d= %d, p= %d, lra= %.3f, noise= %s -- j= %d, alpha= %s, eg= %.10f, time= %.2f' % (d, p, lra, '{:.0e}'.format(noise), 0, '{:.0e}'.format(0), eg_0, 0) )
    '''-------------------------------------'''

    for j in range(1,m_max+1):
        alpha = j/x_scale
        '''Sampling'''
        xj = np.random.randn(d)
        '''Teacher'''
        #################################
        yjt  = ft(beta_t, xj) + noise_sqt*np.random.randn()

        '''Student'''
        #####################################
        h[:] = 0.
        for u in range(0,p):
            for v in range(0,d):
                h[u] += W0[v,u]*xj[v] / sqrt(d)

        yjs = fs(h,a)

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
                        h[u] += W0[v,u]*X_test[v][r] / sqrt(d)
                #####################################
                fs_test = fs(h,a)
                eg_ += 0.5*square(ft_test[r]-fs_test)/m


            eg_avj = eg_
            eg_av_.append(eg_avj)
            alpha_plot_.append(alpha)

            t1 = time.time()
            print('d = %d, p = %d, lra = %.3f, noise= %s -- j = %d, alpha = %s, eg = %.10f, time = %.2f' % (d, p, lra, '{:.0e}'.format(noise), j, '{:.0e}'.format(alpha), eg_avj,t1-t0) )
            t0 = t1

            if np.abs(eg_avj-eg_avj_old) < tol:
                tol_string = True
                break
            eg_avj_old = eg_avj

            if save_j and j < m_max:
                log_alpha_save = np.where(savelist == j)[0][0]
                if save_log:
                  id = 'LOGsave_alpscale_%s_alpha1e%d_%s_eg_af_W0_a0_alpha.npz' % (alpha_scale, log_alpha_save, save_key)
                  dict_save = {'eg':np.array(eg_av_), 'af': np.array(a), 'W0': np.array(W0), 'a0': np.array(a0), 'alpha':np.array(alpha_plot_)}
                  file_path = file_path_id + '/' + id
                  np.savez(file_path, **dict_save)

    if tol_string:
        alpha_final = j/x_scale
        print('Terminating: | Delta(eg)| < %s (tol)' %  '{:.0e}'.format(tol) )
        id = 'FINALsave_tol%s_alpscale_%s_alpha%s_%s_eg_af_W0_a0_alpha.npz' % ('{:.0e}'.format(tol), alpha_scale, '{:.0e}'.format(alpha_final), save_key)
        dict_save = {'eg':np.array(eg_av_), 'af': np.array(a), 'W0': np.array(W0), 'a0': np.array(a0),'alpha':np.array(alpha_plot_)}
    else:
        print('Terminating: alpha = %s (alpha_max)' %  '{:.0e}'.format(alpha_max) )
        id = 'FINALsave_max_alpscale_%s_alpha1e%d_%s_eg_af_W0_a0_alpha.npz' % (alpha_scale, np.log10(alpha_max), save_key)
        dict_save = {'eg':np.array(eg_av_), 'af': np.array(a), 'W0': np.array(W0), 'a0': np.array(a0), 'alpha':np.array(alpha_plot_)}

    file_path = file_path_id + '/' + id
    np.savez(file_path, **dict_save)

    return np.array(alpha_plot_), np.array(eg_av_), np.array(a),  np.array(a0)
