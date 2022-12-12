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

#cdef double f_old(double[:] loc_field):
#    '''Model'''
#    cdef double phi =0.
#    cdef unsigned int j, L= loc_field.shape[0]
#    for j in range(0,L):
#        phi += erf( loc_field[j]/ sqrt(2) )
#    return phi


cdef double fs(double[:] loc_field, double[:] a):
  '''Model'''
  cdef double phi =0.
  cdef unsigned int j, L= loc_field.shape[0]
  for j in range(0,L):
    phi += a[j]*erf( loc_field[j]/ sqrt(2) )
  return phi


cdef double ft(double[:] loc_field, double[:] a):
  '''Function to be fitted'''
  '''Linear teacher for now'''
  cdef double phi =0.
  cdef unsigned int j, L= loc_field.shape[0]
  for j in range(0,L):
    phi += a[j]*loc_field[j]
  return phi


cpdef cython_fit(long m_max, bint norm, double[:,:] W, double[:,:] Wt, double[:] a, double[:] at, double lrW, double lra, double noise, bint linear_t, bint train_a, bint svd_only, double x_scale, double tol, double alpha_max, plotlist, savelist, save_log, file_path_id, alpha_scale, save_key):
    cdef unsigned int j, q, r, r_ , u, v, j1, j2, j3, s, k = Wt.shape[1], p = W.shape[1], d = W.shape[0], m = 1000
    cdef double alpha, eg_avj, yjt, yjs, fs_test, ns, nt, eg_0= 0., eg_avj_old= 0., eg_, pi= acos(-1), noise_sqt = sqrt(noise), wql, st_fac

    cdef double *arr = <double*>malloc(d * sizeof(double))
    cdef double[:] xj = <double[:d]>arr

    cdef double *arr2 = <double*>malloc(d * m * sizeof(double))
    cdef double[:,:] X_test = <double[:d,:m]>arr2

    cdef double *arr3 = <double*>malloc(p * sizeof(double))
    cdef double[:] h = <double[:p]>arr3

    cdef double *arr4 = <double*>malloc(k * sizeof(double))
    cdef double[:] ht = <double[:k]>arr4

    cdef double *arr5 = <double*>malloc(m * sizeof(double))
    cdef double[:] ft_test = <double[:m]>arr5

    cdef list eg_av_=[], alpha_plot_=[]
    cdef str tol_string= False

    cdef double *arr6 = <double*>malloc(p * sizeof(double))
    cdef double[:] h0 = <double[:p]>arr6

    cdef double *arr7 = <double*>malloc(p * sizeof(double))
    cdef double[:] h0_test = <double[:p]>arr7


    cdef double *arr8 = <double*>malloc(d * d * sizeof(double))
    cdef double[:,:] U0 = <double[:d,:d]>arr8


    cdef double *arr9 = <double*>malloc(p * p * sizeof(double))
    cdef double[:,:] V0 = <double[:p,:p]>arr9

    cdef double *arr10 = <double*>malloc(p * sizeof(double))
    cdef double[:] sv0 = <double[:p]>arr10

    cdef double *arr11 = <double*>malloc(p * sizeof(double))
    cdef double[:] sv = <double[:p]>arr11

    cdef double *arr12 = <double*>malloc(p * sizeof(double))
    cdef double[:] hu = <double[:p]>arr12

    '''Normalization of the committee'''
    if norm:
        ns = 1./p
        nt = 1./k
    else:
        ns = 1.
        nt = 1.

    '''Test set'''
    X_test = np.random.randn(d, m)
    ft_test[:] = 0.
    for r in range(0,m):
        ht[:] = 0.
        for u in range(0,k):
            for v in range(0,d):
                ht[u] += Wt[v,u]*X_test[v][r] / sqrt(d)
        ft_test[r] = ft(ht, at)*nt if linear_t else fs(ht, at)*nt


    W0 = np.copy(W)
    a0 = np.copy(a)

    W0T = W0.T
    t0 = time.time()

    if svd_only:
      '''SVD decomposition at the initialization'''
      U0, sv0, V0 = np.linalg.svd(W0, full_matrices=True)
      V0 = np.copy(V0.T)
      sv = np.copy(sv0)

    '''Generalization error at initialization'''
    for r in range(0,m):
        h[:] = 0.
        h0_test[:] = 0.
        for u in range(0,p):
            for v in range(0,d):
                h[u] += W[v,u]*X_test[v][r] / sqrt(d)
                h0_test[u] += W0[v,u]*X_test[v][r] / sqrt(d)
        #####################################
        fs_test = fs(h,a)*ns
        eg_0 += 0.5*square(ft_test[r]-fs_test)/m

    eg_av_.append(eg_0)
    alpha_plot_.append(0.)
    print('d= %d, k= %d, p= %d, lrW= %.3f, lra= %.3f, noise= %s -- j= %d, alpha= %s, eg= %.10f, time= %.2f' % (d, k, p, lrW, lra, '{:.0e}'.format(noise), 0, '{:.0e}'.format(0), eg_0, 0) )
    '''-------------------------------------'''

    for j in range(1,m_max+1):
        alpha = j/x_scale
        '''Sampling'''
        xj = np.random.randn(d)
        '''Teacher'''
        #################################
        ht[:] = 0.
        for u in range(0,k):
            for v in range(0,d):
                ht[u] += Wt[v,u]*xj[v] / sqrt(d)
        ##################################
        yjt  = ft(ht, at)*nt + noise_sqt*np.random.randn() if linear_t else fs(ht, at)*nt + noise_sqt*np.random.randn()

        '''Student'''
        #####################################
        h[:] = 0.
        h0[:] = 0.
        #'''Initializing uT x for r=1,...p '''
        #hu[:] = 0.

        for u in range(0,p):
            for v in range(0,d):
                h[u] += W[v,u]*xj[v] / sqrt(d)
                h0[u] += W0[v,u]*xj[v] / sqrt(d)
                #'''Calculating uT x for r=1,...,p'''
                #hu[u] += W[v,u]*xj[v] /sqrt(d)
        #####################################
        yjs = fs(h,a)*ns

        #'''Gradient (vanilla version)'''
        #for s in range(0,p):
        #    for q in range(0,d):
        #        W[q,s] += ns*(yjt - yjs)*lrW*exp(-square(h[s])/2.)*sqrt(2./pi)*xj[q]/sqrt(d)


        #'''Gradient on the singular values'''
        #for r in range(0,p):
        #    st_fac = 0.
        #    for j1 in range(0,p):
        #        st_fac += a[j1]*np.exp(-square(h[j1])/2.)*sqrt(2./np.pi)*V0[j1,r]
        #    sv[r] += ns*(yjt - yjs)*lrW*hu[r]*st_fac

        #'''Gradient on the a's'''
        #for r in range(0,p):
        #    a[r] += ns*(yjt - yjs)*lra*erf( h[r]/sqrt(2) )

        #'''Reconstruct W'''
        #for q in range(0,d):
        #    for l in range(0,p):
        #        wql = 0.
        #        for r_ in range(0,p):
        #            wql += U0[q,r_]*sv[r_]*V0[l,r_]
        #    W[q,l] = wql

        if svd_only:
          '''Gradient on the singular values'''

          '''Calculate uT x for r=1,...,p'''
          hu[:] = 0.
          for u in range(0,p):
              for v in range(0,d):
                  hu[u] += W[v,u]*xj[v] /sqrt(d)

          '''Update the singular values'''
          for r in range(0,p):
              st_fac = 0.
              for j1 in range(0,p):
                  st_fac += a[j1]*np.exp(-square(h[j1])/2.)*sqrt(2./np.pi)*V0[j1,r]
              sv[r] += ns*(yjt - yjs)*lrW*hu[r]*st_fac

          '''Reconstruct W'''
          for q in range(0,d):
              for l in range(0,p):
                wql = 0.
                for r_ in range(0,p):
                    wql += U0[q,r_]*sv[r_]*V0[l,r_]
              W[q,l] = wql

        else:
          '''Gradient (vanilla version)'''
          for s in range(0,p):
            for q in range(0,d):
              W[q,s] += ns*(yjt - yjs)*lrW*exp(-square(h[s])/2.)*sqrt(2./pi)*xj[q]/sqrt(d)


        if train_a:
          '''Gradient on the a's'''
          for r in range(0,p):
            a[r] += ns*(yjt - yjs)*lra*erf( h[r]/sqrt(2) )

        plot_j = j in plotlist
        save_j = j in savelist

        #print(plot_j)

        if plot_j:
            '''Generalization error'''
            eg_ = 0.
            for r in range(0,m):
                h[:] = 0.
                for u in range(0,p):
                    for v in range(0,d):
                        h[u] += W[v,u]*X_test[v][r] / sqrt(d)
                #####################################
                fs_test = fs(h,a)*ns
                eg_ += 0.5*square(ft_test[r]-fs_test)/m

            eg_avj = eg_
            eg_av_.append(eg_avj)
            alpha_plot_.append(alpha)

            t1 = time.time()
            print('d = %d, k = %d, p = %d, lrW = %.3f, lra = %.3f, noise= %s -- j = %d, alpha = %s, eg = %.10f, time = %.2f' % (d, k, p, lrW, lra, '{:.0e}'.format(noise), j, '{:.0e}'.format(alpha), eg_avj,t1-t0) )
            t0 = t1

            if np.abs(eg_avj-eg_avj_old) < tol:
                tol_string = True
                break
            eg_avj_old = eg_avj

            if save_j and j < m_max:
                log_alpha_save = np.where(savelist == j)[0][0]
                if save_log:
                  id = 'LOGsave_alpscale_%s_alpha1e%d_%s_eg_Wt_at_Wf_af_W0_a0_alpha.npz' % (alpha_scale, log_alpha_save, save_key)
                  dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'at': np.array(at), 'Wf': np.array(W), 'af': np.array(a), 'W0': np.array(W0), 'a0': np.array(a0), 'alpha':np.array(alpha_plot_)}
                  file_path = file_path_id + '/' + id
                  np.savez(file_path, **dict_save)

    if tol_string:
        alpha_final = j/x_scale
        print('Terminating: | Delta(eg)| < %s (tol)' %  '{:.0e}'.format(tol) )
        id = 'FINALsave_tol%s_alpscale_%s_alpha%s_%s_eg_Wt_at_Wf_af_W0_a0_alpha.npz' % ('{:.0e}'.format(tol), alpha_scale, '{:.0e}'.format(alpha_final), save_key)
        dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'at': np.array(at), 'Wf': np.array(W), 'af': np.array(a), 'W0': np.array(W0), 'a0': np.array(a0),'alpha':np.array(alpha_plot_)}
    else:
        print('Terminating: alpha = %s (alpha_max)' %  '{:.0e}'.format(alpha_max) )
        id = 'FINALsave_max_alpscale_%s_alpha1e%d_%s_eg_Wt_at_Wf_af_W0_a0_alpha.npz' % (alpha_scale, np.log10(alpha_max), save_key)
        dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'at': np.array(at), 'Wf': np.array(W), 'af': np.array(a), 'W0': np.array(W0), 'a0': np.array(a0), 'alpha':np.array(alpha_plot_)}

    file_path = file_path_id + '/' + id
    np.savez(file_path, **dict_save)

    return np.array(alpha_plot_), np.array(eg_av_), np.array(W),  np.array(W0)
