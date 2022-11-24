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

cdef double f(double[:] loc_field):
    '''Model'''
    cdef double phi =0.
    cdef unsigned int j, L= loc_field.shape[0]
    for j in range(0,L):
        phi += erf( loc_field[j]/ sqrt(2) )
    return phi


cpdef cython_fit(long m_max, bint norm, double[:,:] W, double[:,:] Wt, double lr, double noise, double x_scale, bint lazy, double tol, double alpha_max, plotlist, savelist, save_log, file_path_id, alpha_scale, save_key):
    cdef unsigned int j, q, r, u, v, j1, j2, j3, s, k = Wt.shape[1], p = W.shape[1], d = W.shape[0], m = 12
    cdef double alpha, eg_avj, yjt, yjs, fs_test, ns, nt, lr_, eg_0= 0., eg_avj_old= 0., eg_, pi= acos(-1), noise_sqt = sqrt(noise)

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

    '''Normalization of the committee'''
    if norm:
        ns = 1./p
        nt = 1./k
    else:
        ns = 1.
        nt = 1.

    print('d= ', d)
    print('p= ', p)
    print('k= ', k)

    '''Learning rate scale (from the updated eqs do not change)'''
    lr_= ns*(lr/sqrt(d))

    '''Test set'''
    X_test = np.random.randn(d, m)
    ft_test[:] = 0.
    for r in range(0,m):
        ht[:] = 0.
        for u in range(0,k):
            for v in range(0,d):
                ht[u] += Wt[v,u]*X_test[v][r] / sqrt(d)
        ft_test[r] = f(ht)*nt


    W0 = np.copy(W)
    W0T = W0.T
    t0 = time.time()

    '''SVD decomposition at the initialization'''
    U0, sv0, V0 = np.linalg.svd(W0, full_matrices=True)
    V0 = np.copy(V0.T)


    '''Generalization error at initialization'''
    for r in range(0,m):
        h[:] = 0.
        h0_test[:] = 0.
        for u in range(0,p):
            for v in range(0,d):
                h[u] += W[v,u]*X_test[v][r] / sqrt(d)
                h0_test[u] += W0[v,u]*X_test[v][r] / sqrt(d)
        #####################################
        fs_test = f(h)*ns
        eg_0 += 0.5*square(ft_test[r]-fs_test)/m

    eg_av_.append(eg_0)
    alpha_plot_.append(0.)
    print('d= %d, k= %d, p= %d, lr= %.3f, noise= %s -- j= %d, alpha= %s, eg= %.10f, time= %.2f' % (d, k, p, lr, '{:.0e}'.format(noise), 0, '{:.0e}'.format(0), eg_0, 0) )
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
        yjt  = f(ht)*nt + noise_sqt*np.random.randn()

        '''Student'''
        #####################################
        h[:] = 0.
        h0[:] = 0.
        for u in range(0,p):
            for v in range(0,d):
                h[u] += W[v,u]*xj[v] / sqrt(d)
                h0[u] += W0[v,u]*xj[v] / sqrt(d)
        #####################################
        yjs = f(h)*ns

        '''Gradient'''
        for s in range(0,p):
            for q in range(0,d):
                W[q,s] += (yjt - yjs)*lr_*exp(-square(h[s])/2.)*sqrt(2./pi) *xj[q]

        #'''Gradient on the singular values'''
        #for r in range(0,p):
        #    ur_dot_x = 0.
        #    for l in range(0,d):
        #      ur_dot_x += U0[l,r]*xj[l]

        #    st_fac = 0.
        #    for j in range(0,p):
        #      st_fac += np.exp(-square(h[j])/2.)*sqrt(2./np.pi)*V0[j,r]

        #    s[r] += (yjt - yjs)*lr_*st_fac*ur_dot_x

        #'''Reconstruct W'''
        #'''just a start: it will be super slow'''
        # create m x n Sigma matrix
        #Sigma = np.zeros((W0.shape[0], W0.shape[1]))
        # populate Sigma with n x n diagonal matrix
        #Sigma[:W0.shape[1], :W0.shape[1]] = np.diag(s)
        # reconstruct matrix
        #W = U0.dot(Sigma.dot(V0))


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
                fs_test = f(h)*ns
                eg_ += 0.5*square(ft_test[r]-fs_test)/m

            eg_avj = eg_
            eg_av_.append(eg_avj)
            alpha_plot_.append(alpha)

            t1 = time.time()
            print('d = %d, k = %d, p = %d, lr = %.3f, noise= %s -- j = %d, alpha = %s, eg = %.10f, time = %.2f' % (d, k, p, lr, '{:.0e}'.format(noise), j, '{:.0e}'.format(alpha), eg_avj,t1-t0) )
            t0 = t1

            if np.abs(eg_avj-eg_avj_old) < tol:
                tol_string = True
                break
            eg_avj_old = eg_avj

            if save_j and j < m_max:
                log_alpha_save = np.where(savelist == j)[0][0]
                if save_log:
                  id = 'LOGsave_alpscale_%s_alpha1e%d_%s_eg_Wt_Wf_W0_alpha.npz' % (alpha_scale, log_alpha_save, save_key)
                  dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_)}
                  file_path = file_path_id + '/' + id
                  np.savez(file_path, **dict_save)

    if tol_string:
        alpha_final = j/x_scale
        print('Terminating: | Delta(eg)| < %s (tol)' %  '{:.0e}'.format(tol) )
        id = 'FINALsave_tol%s_alpscale_%s_alpha%s_%s_eg_Wt_Wf_W0_alpha.npz' % ('{:.0e}'.format(tol), alpha_scale, '{:.0e}'.format(alpha_final), save_key)
        dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_)}
    else:
        print('Terminating: alpha = %s (alpha_max)' %  '{:.0e}'.format(alpha_max) )
        id = 'FINALsave_max_alpscale_%s_alpha1e%d_%s_eg_Wt_Wf_W0_alpha.npz' % (alpha_scale, np.log10(alpha_max), save_key)
        dict_save = {'eg':np.array(eg_av_), 'Wt': np.array(Wt), 'Wf': np.array(W), 'W0': np.array(W0), 'alpha':np.array(alpha_plot_)}

    file_path = file_path_id + '/' + id
    np.savez(file_path, **dict_save)

    return np.array(alpha_plot_), np.array(eg_av_), np.array(W),  np.array(W0)
