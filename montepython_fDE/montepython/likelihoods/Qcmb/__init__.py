from montepython.likelihood_class import Likelihood
from numpy import matrix, dot, exp, log


class Qcmb(Likelihood):
    def loglkl(self, cosmo, data):
        ths =   cosmo.theta_s_100()/100
        ob  =   cosmo.omega_b()
        ocb =   ob  +   cosmo.Omega0_cdm()*cosmo.h()**2
        diffvec = matrix([x-mu for x, mu in zip([ths, ob, ocb], self.centre)])
        minusHessian = matrix(self.covmat).I
        return -0.5 * (dot(diffvec, dot(minusHessian, diffvec.T)))
