import numpy as np
from sklearn.preprocessing import normalize
import os
from rfcy import cython_fit
np.random.seed(0)

class rf():
    def __init__(self, d, p, k):
        self.d = d
        self.p = p
        self.k = k

    def set_teacher(self, beta_t= None):

        self.beta_t = np.sqrt(self.d)* normalize(np.random.randn(self.d)[:,np.newaxis], axis=0).ravel()

        return self.beta_t


    def set_Theta0(self, W0= None, a0= None):

        self.W0 =  np.sqrt(self.d)*normalize(np.random.randn(self.d, self.p), axis=1, norm='l2') if W0 is None else W0
        self.a0 = np.sqrt(self.p)*normalize(np.random.randn(self.p)[:,np.newaxis], axis=0).ravel()

        return self.W0, self.a0


    def fit(self, alpha_max, kappa, delta, lra0=None, noise= 0., tol=1e-10, save_log= False, save_folder= 'results/rf/', save_key= ''):

        print('- - - RANDOM FEATURES MODEL - - -')
        if noise > 0:
            print('Linear teacher with additive output noise: y_t = beta x/sqrt(d) + sqrt(noise)*xi ; noise= %s' % '{:.0e}'.format(noise))
        else:
            print('Noiseless linear teacher : y_t = beta x/sqrt(d)')

        print('Scaling:')

        if (kappa + delta) > 0:
            print('GREEN region: kappa + delta > 0')
            x_scale = self.d**(1.+kappa+delta)
            alpha_scale = 'd%.4f' % (1.+kappa+delta)
            region = 'green'

        if (kappa + delta) == 0:
            print('BLUE line: kappa + delta = 0')
            x_scale = self.d**(1.+kappa+delta)
            alpha_scale = 'd%.4f' % (1.+kappa+delta)
            region = 'blue'

        if -delta  > kappa and -delta < (kappa+1.)/2:
            print('ORANGE region: kappa < -delta < (kappa+1)/2')
            x_scale = self.d**(1.+2*(kappa+delta))
            alpha_scale = 'd%.4f' % (1.+2*(kappa+delta))
            region = 'orange'

        lra = lra0 / (self.d**(delta))

        print('alpha_scale: ',alpha_scale)

        m_max = int(alpha_max*x_scale)
        log10_alpha_max = int(np.log10(alpha_max))
        savelist = self.savelog_list(log_x_max= log10_alpha_max, scale= x_scale)
        plotlist = self.xlog_scale(log_x_max= log10_alpha_max, scale= x_scale)

        dformat = '{:0'+str(int(round(np.log10(self.d))))+'}'
        print_d = dformat.format(self.d)
        folder_id = '%s_d%s_p%s_k%s_kappa%.5f_delta%.5f_lra0%.5f_noise%s' % (region, print_d, '{:03}'.format(self.p), '{:03}'.format(self.k), kappa, delta, lra0, '{:.0e}'.format(noise) )

        file_path_id = save_folder + folder_id
        isExist = os.path.exists(file_path_id)
        if not isExist:
            os.makedirs(file_path_id)

        alphaf, egf, Wf, W0_ = cython_fit(m_max, self.W0, self.beta_t, self.a0, lra, noise, x_scale, tol, alpha_max, plotlist, savelist, save_log, file_path_id, alpha_scale, save_key)

        return alphaf, egf, Wf, W0_



    def xlog_scale(self, log_x_max, scale, log_base=10):
        '''Logaritmic scale up to log_alpha_max'''

        bd_block = np.arange(0, log_base**2, log_base) + log_base
        bd_block = bd_block[0:-1]
        xlog = np.tile(bd_block, log_x_max)

        xlog[(log_base-1) : 2*(log_base-1)] = log_base*xlog[(log_base-1) : 2*(log_base-1)]

        for j in range(1, log_x_max - 1):
            xlog[(j+1)*(log_base-1) : (j+2)*(log_base-1)] = log_base*xlog[  j*(log_base-1) :  (j+1)*(log_base-1)  ]

        xlog = np.insert(xlog, 0,  np.arange(1,log_base), axis=0)
        xlog = np.insert(xlog, len(xlog),log_base**(log_x_max+1), axis=0)

        jlog = (xlog*scale).astype(int)

        return jlog


    def savelog_list(self, log_x_max, scale):
        '''Logaritmic scale up to log_alpha_max'''
        xlog = np.logspace(0, log_x_max, log_x_max+1, endpoint=True).astype(int)
        save_xlog = (xlog*scale).astype(int)
        return save_xlog
