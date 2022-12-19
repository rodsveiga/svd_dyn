import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
from scipy.linalg import diagsvd
from scipy.stats import semicircular
from scipy.stats import ortho_group
import os
from svdcy import cython_fit
np.random.seed(0)

class svd():
    def __init__(self, d, p):
        self.d = d
        self.p = p


    def set_teacher(self, beta_t= None):
        self.beta_t = np.sqrt(self.d)* normalize(np.random.randn(self.d)[:,np.newaxis], axis=0).ravel()
        return self.beta_t


    def set_Theta0(self):
        U0  = ortho_group.rvs(self.d)
        sv0 = np.abs(semicircular.rvs(size=self.p))
        V0 = ortho_group.rvs(self.p)
        order_p = np.argsort(-sv0)
        order_d = np.concatenate((order_p, np.arange(self.p,self.d)))

        self.U0 = U0[:, order_d]
        self.sv0 = sv0[order_p]
        self.V0t = (V0[:, order_p]).T

        self.a0 = np.sqrt(self.p)*normalize(np.random.randn(self.p)[:,np.newaxis], axis=0).ravel()

        return self.U0, self.sv0, self.V0t, self.a0


    def fit(self, t_max, kappa, delta, lrRF=1., alpha=0, noise= 0., tol=0., save_log= False, save_folder= 'results/', save_key= ''):

        print('- - - - SVD MODEL - - - -')
        if noise > 0:
            print('Linear teacher with additive output noise: y_t = beta x/sqrt(d) + sqrt(noise)*xi ; noise= %s' % '{:.0e}'.format(noise))
        else:
            print('Noiseless linear teacher : y_t = beta x/sqrt(d)')


        print('--- Phase diagram ---')
        if (kappa + delta) > 0:
            print('GREEN region: kappa + delta > 0')
            x_scale = self.d**(1.+kappa+delta)
            t_scale = 'd%.4f' % (1.+kappa+delta)
            region = 'green'

        if (kappa + delta) == 0:
            print('BLUE line: kappa + delta = 0')
            x_scale = self.d**(1.+kappa+delta)
            t_scale = 'd%.4f' % (1.+kappa+delta)
            region = 'blue'

        if -delta  > kappa and -delta < (kappa+1.)/2:
            print('ORANGE region: kappa < -delta < (kappa+1)/2')
            x_scale = self.d**(1.+2*(kappa+delta))
            t_scale = 'd%.4f' % (1.+2*(kappa+delta))
            region = 'orange'

        print('Time scaling: ',t_scale)
        print('--- Learning rates W / a ---')
        print('lrW = alpha x lrRF')
        lrW0 = alpha*lrRF
        lrW = lrW0 / (self.d**(delta))

        m_max = int(t_max*x_scale)

        log10_t_max = int(np.log10(t_max))
        savelist = self.savelog_list(log_x_max= log10_t_max, scale= x_scale)
        plotlist = self.xlog_scale(log_x_max= log10_t_max, scale= x_scale)
        dformat = '{:0'+str(int(round(np.log10(self.d))))+'}'
        print_d = dformat.format(self.d)

        folder_id = '%s_d%s_p%s_kappa%.5f_delta%.5f_lrRF%.5f_alpha%.5f_noise%s' % (region, print_d, '{:03}'.format(self.p), kappa, delta, lrRF, alpha, '{:.0e}'.format(noise))

        file_path_id = save_folder + folder_id
        isExist = os.path.exists(file_path_id)
        if not isExist:
            os.makedirs(file_path_id)

        alphaf, egf, Wf = cython_fit(m_max, self.U0, self.sv0, self.V0t, self.beta_t, self.a0, lrW, lrRF, noise, x_scale, tol, t_max, plotlist, savelist, save_log, file_path_id, t_scale, save_key)

        return alphaf, egf, Wf


    def xlog_scale(self, log_x_max, scale, log_base=10):
        '''Logaritmic scale up to log_t_max'''
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
        '''Logaritmic scale up to log_t_max'''
        xlog = np.logspace(0, log_x_max, log_x_max+1, endpoint=True).astype(int)
        save_xlog = (xlog*scale).astype(int)
        return save_xlog
