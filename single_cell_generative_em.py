# ** Imports ** #
import copy
import numpy as np
from scipy.misc import logsumexp as lse
import numpy.random as rnd
import scipy.stats as stts
from sklearn.manifold import t_sne
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import mmread
from sklearn.cluster import KMeans


# ** matplotlib styles ** #
# mpl.style.use('seaborn')
mpl.style.use('seaborn-pastel')


# ** Constants ** #
EXP_NOISE = .01
EPS = 1e-10
INPUT_PATH = 'muris-Lung-10X_P7_9.mtx'
K = [6]


# ** Model of single cell transcript counts class ** #
class SingleCellExpressionModel(object):

    def __init__(self, C=100, G=500, T=5, s=0.1, p_mean=0.05, mu=None, tau=None, p=None):
        """
        :param C: number of cells
        :param G: number of genes
        :param T: number of cell types
        :param s: gene sparseness, only relevant if mu is not given
        :param p_mean: average capture rate per cell, only relevant if p is not given
        :param mu: an optioanl GxT matrix indicating expected expression from each gene in each type. Overrides G, T, s
        :param tau: an optional positive vector of length T with the prior probability for each type. Overrides T.
        :param p: an optional vector of length C in [0,1], indicating capture probability per cell. Overrides C, p_mean
        """
        if mu is not None:
            G, T = mu.shape
        if tau is not None:
            assert all(tau > 0)
            T = len(tau)
        if p is not None:
            assert all(p > 0) and all(p <= 1)
            C = len(p)
        self.C = C
        self.G = G
        self.T = T
        self.mu = mu
        if mu is None:
            self.mu = 10 ** (rnd.normal(size=(G, T)) * 1.5)  # log-normal expression distribution
            nG = min(G, max([5, round(G / 10)]))  # 10% of genes will have a common mean in all types
            cg_idx = rnd.permutation(G)[:nG]
            self.mu[cg_idx, :] = np.tile(self.mu[cg_idx, 0], (T, 1)).T
            self.mu[rnd.random(self.mu.shape) < s] = EXP_NOISE  # dotted with unexpressed genes
        self.p = rnd.beta(100 * p_mean, 100 * (1 - p_mean),
                          size=C) if p is None else p  # sample p's from a beta distribtuion around p_mean
        if tau is None: tau = .1 + rnd.random(size=T)
        tau /= tau.sum()
        self.log_tau = np.log(tau)

    def __sub__(self, other):
        """
        Calculate the parameter difference norm between two models
        :param other: another model
        :return: ||theta1-theta2||
        """
        return np.linalg.norm(self.ravel() - other.ravel())

    def ravel(self):
        """
        Vectorize model parameters
        :return: [theta; p; mu]
        """
        return np.hstack((np.exp(self.log_tau).ravel(), self.p.ravel(), self.mu.ravel()))

    def sample(self):
        """
        Sample a data set from this model

        :return:
            x - A sampled observed matrix [GxC] of molecular counts
            y - A sampled hidden matrix [GxC] of the actual molecular counts, used to generate x
            t - A sampled hidden vector [C] of types per cell, used to generate y
        """
        t = np.nonzero(rnd.multinomial(1, np.exp(self.log_tau), self.C))[1]
        y = rnd.poisson(self.mu[:, t])
        x = rnd.binomial(y, np.tile(self.p, [self.G, 1]))
        return x, y, t

    def _type_likelihood(self, x):
        """
        Calculate data probability per type per cell
        :param x: Observed matrix [GxC] of molecular counts
        :return: log[Pr(X_c|T_c=t,model)]
        """
        x = np.tile(x, [self.T, 1, 1])
        mu = np.transpose(np.tile(self.mu, [self.C, 1, 1]))
        p = np.tile(self.p, [self.T, self.G, 1])

        poi_mass_func = stts.poisson.logpmf(x, mu * p)
        return poi_mass_func.sum(1).T + np.tile(self.log_tau, [self.C, 1])

    def loglikelihood(self, x, return_tl=False):
        """
        Compute the log likelihood of x, given this model
        :param x: Observed matrix [GxC] of molecular counts
        :param return_tl: whether type probability per cell should also be returned
        :return: Pr(X|model)[, log[Pr(X_c|T_c=t,model)], if requested]
        """
        tl = self._type_likelihood(x)
        ll = np.sum(lse(tl, 1))
        return (ll, tl) if return_tl else ll

    def type_posterior(self, x, tl=None):
        """
        Calculate the log-posterior probability of types
        :param x: Observed matrix [GxC] of molecular counts
        :return tl: type likelihood. If not given, calculated from x, if given - x is ignored.
        """
        if tl is None: tl = self._type_likelihood(x)
        tp = tl - np.reshape(lse(tl, axis=1), [self.C, 1])
        return tp

    def _q_argmax(self, tp, x, learn_params):
        """
        Calculate the parameters maximizing the Q function (lower bound)
        :param tp: Type posterior probability
        :param x: Observed matrix [GxC] of molecular counts
        :param learn_params: Which parameters should the model improve: 'mu', 'tau', 'p', or any combination of these.
                             Default is ['mu', 'tau', 'p']. Returns the old version of parameters not in this list.
        :return: A tuple of new parameters (log(tau), p, mu)
        """
        tau_ = self.log_tau
        mu_ = self.mu
        p_ = self.p
        tp_exp = np.exp(tp)  # was already normalized so OK to exponentiate

        if 'tau' in learn_params:
            tau_ = lse(tp, 0) + np.log(EPS)  # logsumexp the probabilties over all cells, and regularize
            tau_ = tau_ - lse(tau_)  # normalize in log space

        if 'p' in learn_params:
            Xcg = np.sum(x,axis = 0)
            muTc = np.sum(mu_, axis =0).reshape((self.T,1))
            p_ = Xcg / (Xcg + (1-p_) * np.sum((tp_exp*muTc.T),1))

        if 'mu' in learn_params:
            x_tile = np.tile(x, [self.T, 1, 1])
            mu_tile = np.transpose(np.tile(mu_, [self.C, 1, 1]))
            comp_p_tile = np.tile((1 - self.p), [self.T, self.G, 1])
            Wtc_tile = np.transpose(np.tile(tp_exp, [self.G, 1, 1]), [2,0,1])
            Wtc_norm_fact = tp_exp.sum(0).reshape((1, self.T)) + EPS
            mu_ = (Wtc_tile * (mu_tile * comp_p_tile + x_tile)).sum(2).T / Wtc_norm_fact

        return tau_, p_, mu_

    def learn(self, x, maxiter=1000, tolerance=0.1, report=None, noise_amp=0, noise_decay_rate=0.1, learn_params=None):
        """
        Fit this model to x using an em algorithm
        :param x: Observed matrix [GxC] of molecular counts
        :param maxiter: maximum number of iterations to perform
        :param tolerance: likelihood improvement tolerance under which the algorithm halts
        :param report: a function handle that gets:
                       (likelihood_history, type likelihood per cell, current_model) and can print/plot
                       statistics while algorithm is running
                       e.g.:
                           def print_iter(ll, tp, m):
                               print('At iter %i, ll: %.2f (%.2f)' % (it, ll[-1], ll[-1] - ll[-2]))
        :param noise_amp: It's possible to add some noise to the learning process. The noise added (N) behaves like:
                            N = A * x^-r
                          Where A is noise_map, x is the current iteration number, and r is the noise_decay_rate.
        :param noise_decay_rate: See noise_amp.
        :param learn_params: Which parameters should the model improve: 'mu', 'tau', 'p', or any combination of these.
                             Default is ['mu', 'tau', 'p'].

        :return: the log likelihood history
        """
        if not learn_params: learn_params = ['mu', 'tau', 'p']
        ll = [-float('inf')]
        for it in range(maxiter):
            lli, tl = self.loglikelihood(x, return_tl=True)
            ll.append(lli)

            # (noisy) E step
            tl += np.tile(self.log_tau, [self.C, 1])
            n = noise_amp * ((it + 1) ** -noise_decay_rate)
            if n > 0:
                tl = np.minimum(0, (tl * (1 + rnd.normal(size=tl.shape) * n)))
            tp = self.type_posterior(x, tl)

            # check halt criteria and report
            if abs(ll[-1] - ll[
                -2]) < tolerance: break  # the abs is just for numeric noise, the difference should always be positive!
            if report: report(ll, tp, self)

            # M step
            self.log_tau, self.p, self.mu = self._q_argmax(tp, x, learn_params)

        return ll


# ** Needed only for Appendix A Example ** #
def confusion_matrix(x, y):
    u = np.unique(np.hstack((x.ravel(), y.ravel())))
    cm = np.zeros((len(u), len(u)))
    for i, xi in enumerate(u):
        for j, yj in enumerate(u):
            cm[i, j] = np.bitwise_and(x == xi, y == yj).sum()
    return cm


# ** Helper functions ** #
def process_data(path):
    _data = mmread(path)
    _data = np.asarray(_data.todense())
    return _data[_data.sum(1) > 0,:]


# ** Grapes required for questions ** #
def quest_9(_data):
    """
    Scatter-plot the logarithm of the gene noise
    """
    mean = np.mean(_data, axis=1)
    log_mean = np.log10(mean)
    var_mean = np.sqrt(mean)

    std = np.std(_data, axis=1)
    log_noise = np.log10(std/mean)

    dense = np.vstack([log_mean, log_noise])
    dense = stats.gaussian_kde(dense)(dense)

    log_poi_noise = np.log10(var_mean/mean)

    plt.plot(log_mean, log_poi_noise)
    plt.scatter(log_mean, log_noise, c=dense, s=3)

    plt.ylabel('log of noise (mean/std)')
    plt.xlabel('log of mean')
    plt.show()


def quest_11(_data):
    """
    Plots of question 11 supplementary code (tSNE vs. PCA)
    """
    tSNE = t_sne.TSNE(n_components=2).fit_transform(np.log(1 + _data.T))
    pca = PCA(n_components=2).fit(np.log(1 + _data.T)).transform(np.log(1 + _data.T))
    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    c = stats.gaussian_kde(tSNE.T)(tSNE.T)
    plt.scatter(tSNE[:, 0], tSNE[:, 1], c=c, s=3)
    plt.title('tSNE')
    plt.xlim(np.percentile(tSNE[:, 0], [1, 100]))
    plt.ylim(np.percentile(tSNE[:, 1], [1, 100]))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    c = stats.gaussian_kde(pca.T)(pca.T)
    plt.scatter(pca[:, 0], pca[:, 1], c=c, s=3)
    plt.title('PCA')
    plt.xlim(np.percentile(pca[:, 0], [1, 100]))
    plt.ylim(np.percentile(pca[:, 1], [1, 100]))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def quest_13(_x):
    """
    Sample data for 1000 cells, with 200 genes, and 8 cell types. Cluster the data with k-means (k = 8)
    """
    plt.subplot(121)
    plt.imshow(np.log(_x), cmap="binary", interpolation="nearest")
    plt.ylabel('Genes')
    plt.xlabel('Cells')
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.show()

    plt.subplot(122)
    kmeans = KMeans( n_clusters=8)
    k_pred = kmeans.fit_predict(_x.T)
    plt.imshow(np.log(_x.T[np.argsort(k_pred)].T), cmap="binary", interpolation="nearest")
    plt.ylabel('Genes')
    plt.xlabel('Cells')
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.show()


def quest_14():
    """
    Sample data from two different models and report the likelihood of each dataset given the two different models.
    """
    model1 = SingleCellExpressionModel(C=100, G=50, T=5)
    model2 = SingleCellExpressionModel(C=100, G=50, T=5)

    x1a, _, _ = model1.sample()
    x1b, _, _ = model1.sample()
    x2, _, _ = model2.sample()

    labels = ['x1a|m1', 'x1b|m1', 'x2|m1', 'x1a|m2', 'x1b|m2', 'x2|m2']
    ll = [model1.loglikelihood(x1a), model1.loglikelihood(x1b), model1.loglikelihood(x2), model2.loglikelihood(x1a), model2.loglikelihood(x1b), model2.loglikelihood(x2)]

    plt.bar(range(len(ll)), ll)
    ax = plt.gca()
    ax.set_xticks(range(len(ll)))
    ax.set_xticklabels(labels)
    plt.ylabel('log Likelihood')
    plt.xlabel('Dataset')
    plt.show()


def quest_15():
    """
    Two models (G=30, C=100, T=6). Use one of the models to sample a small dataset. Run the learn function of both
    models with the same dataset.
    """
    m = [SingleCellExpressionModel(C=100, G=30, T=6), SingleCellExpressionModel(C=100, G=30, T=6)]
    _x, _, _ = m[0].sample()
    ll_opt = m[0].loglikelihood(_x)

    # We sample small dataset from model 0, and check on the entire model 1.
    for model in m:
        plt.subplot('221')
        ll1 = copy.deepcopy(model).learn(_x, learn_params='tau')
        plt.plot(ll1)
        plt.title('only tau')
        plt.xlabel('iterations')
        plt.ylabel('log likelihood')

        plt.subplot('222')
        ll2 = copy.deepcopy(model).learn(_x, learn_params='mu')
        plt.plot(ll2)
        plt.title('only mu')
        plt.xlabel('iterations')
        plt.ylabel('log likelihood')

        plt.subplot('223')
        ll3 = copy.deepcopy(model).learn(_x, learn_params='p')
        plt.plot(ll3)
        plt.title('only p')
        plt.xlabel('iterations')
        plt.ylabel('log likelihood')

        plt.subplot('224')
        ll = copy.deepcopy(model).learn(_x)
        plt.plot(ll)
        plt.plot(ll1)
        plt.plot(ll2)
        plt.plot(ll3)
        plt.plot(plt.xlim(), [ll_opt, ll_opt], 'r--')
        plt.title('all')
        plt.xlabel('iterations')
        plt.ylabel('log likelihood')
        plt.legend(['all', 'tau', 'mu', 'p', 'generating model'], loc=1)
        plt.show()

    # Check for 100 repeats of different initial models.
    _x100 = range(100)
    result = []
    for i in _x100:
        m = SingleCellExpressionModel(C=100, G=30, T=6)
        m.learn(_x, maxiter = 100000)
        final = m.loglikelihood(_x)
        result.append(final)
    plt.scatter(_x100, result)
    plt.plot(plt.xlim(), [ll_opt, ll_opt], 'r--')
    plt.show()


def quest_16(_data):
    """
    Pick some K (or several) apply the EM you’ve implemented to the single cell data with the geneset you had selected.
    Compare the results to k-means. Plot the clustering of the different methods on a 2D tSNE embedding and discuss your
    results.
    """
    max_var_index = np.argsort(np.var(_data, axis=1))[-300:]
    _x = _data[max_var_index, :]

    m = SingleCellExpressionModel(C=_x.shape[1], G=_x.shape[0], T=K[0])
    km = KMeans(n_clusters=K[0]).fit(_x.T)
    m.learn(_x, maxiter=100)
    posterior = m.type_posterior(_x)

    # plt.title('tSNE of k-means clustering colored by centroids')
    tsne = t_sne.TSNE(n_components=2).fit_transform(np.log(1 + _x.T))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=km.labels_, s=4)
    plt.xlim(np.percentile(tsne[:, 0], [1, 100]))
    plt.ylim(np.percentile(tsne[:, 1], [1, 100]))
    plt.show()

    # plt.title('tSNE of EM clustering colored by cell types')
    plt.scatter(tsne[:, 0], tsne[:, 1], c=np.argmax(posterior, axis=1), s=4)
    plt.xlim(np.percentile(tsne[:, 0], [1, 100]))
    plt.ylim(np.percentile(tsne[:, 1], [1, 100]))
    plt.show()


if __name__ == '__main__':

    data = process_data(INPUT_PATH)
    
    quest_9(data)
    quest_11(data)

    m_true = SingleCellExpressionModel()
    x, y, t = m_true.sample()

    quest_13(x)

    m_hat = SingleCellExpressionModel()

    def print_iter(ll, tp, m):
        print(ll[-1])

    m_hat.learn(x, maxiter=100, noise_amp=0.5)
    m_hat.learn(x, maxiter=50, noise_amp=0.3)
    m_hat.learn(x, maxiter=100)
    print(sorted(np.exp(m_true.log_tau)))
    print(sorted(np.exp(m_hat.log_tau)))

    quest_14()
    quest_15()
    quest_16(data)

