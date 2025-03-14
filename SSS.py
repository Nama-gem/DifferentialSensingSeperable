import time
from scipy.optimize import minimize, Bounds
from functions import *
from scipy.sparse.linalg import expm

from itertools import combinations

filename = ('/users/rey/raka3858/DifferentialSensingSeperable/data/SSS')

print(filename)

try:
    data = np.load(filename + '.npz')
    S_ = data['S']
    mu = data['mu']
    phi = data['phi']
    F = data['F']
except:
    S_ = np.append(np.arange(1 / 2, 32, 1 / 2), 2 * np.round(2 ** np.arange(4, 9.25, 0.25)).astype(int))
    phi = np.pi / 8 * np.ones(S_.shape)
    mu = S_ ** (- 1 / 3)
    F = np.ones(S_.shape)
    np.savez(filename, S = S_, phi = phi, F = F)


nPhi = int(1E4)

# for i in np.arange(0, len(S_)):
for i in np.arange(73, len(S_)):
# for i in np.arange(60, 0, - 1):

    S = S_[i]

    ts = time.time()

    Bx = np.real(expm(- 1j * np.pi / 2 * sy(S).toarray()))

    # method = 'L-BFGS-B'
    method = 'SLSQP'
    # method = 'BFGS'

    tol = 1E-16

    # basis = np.kron(Bx, Bx)
    # m = np.kron(np.arange(S, - S - 1, - 1), np.ones(int(2 * S + 1))) + np.kron(np.ones(int(2 * S + 1)), np.arange(S, - S - 1, - 1))
    # ind_rho = np.concatenate([np.array(list(combinations(np.argwhere(m == m1).ravel(), 2))) for m1 in np.arange(2 * S - 1, - 2 * S, - 1)])


    # res = minimize(lambda x: F_prod_der(x[0], oat(x[1], S, Bx), oat(x[1], S, Bx), basis, ind_rho),
    #                np.array([phi[i - 1], mu[i - 1]]), jac=True, method=method,
    #                bounds=Bounds([0, 0], [np.pi / 2, 0.25]))

    # res = minimize(lambda x: F_prod_der(x[0], oat(x[1], S, Bx), oat(x[1], S, Bx), basis, ind_rho), np.array([phi[i - 1], mu[i - 1]]), jac = True, method = method,
    #               bounds = Bounds([0, 0], [np.pi / 2, 0.25]), tol = tol)

    # res = minimize(lambda x: F_prod_der(x[0], oat(x[1], S, Bx), oat(x[1], S, Bx), basis, ind_rho), np.array([phi[i + 1], mu[i + 1]]), jac = True, method = method,
    #               bounds = Bounds([0, 0], [np.pi / 2, 0.25]), tol = tol)

    # res = minimize(lambda x: F_prod_der(x[0], oat(x[1], S, Bx), oat(x[1], S, Bx), basis, ind_rho), np.array([phi[i + 1], mu[i + 1]]), jac = True, method = method,
    #               bounds = Bounds([0, 0], [np.pi / 2, mu[np.max([i - 2, 0])]]))

    # res = minimize(lambda x: F_prod_der(x[0], oat(x[1], S, Bx), oat(x[1], S, Bx), basis, ind_rho), np.array([0.39190269, 0.1915599]), jac = True, method = method,
    #               bounds = Bounds([0, 0], [np.pi / 2, 0.25]))

    res = minimize(lambda x: F_prod_fast_der(x[0], oat(x[1], S, Bx), oat(x[1], S, Bx), nPhi, Bx),
                   np.array([np.exp(-0.68877735 - 0.40519728 * np.log(S)), np.exp(-0.65858885-0.7362593 * np.log(S))]),
                   jac = True, method = method, bounds=Bounds([0, 0], [np.pi / 2, 0.25]), tol = tol)

    te = time.time()

    if - res.fun > F[i]:
        print('Improved from ' + str(np.round(dB(F[i]), 4)) + ' to ' + str(np.round(dB(- res.fun), 4)) + ' (dB)')

        phi[i] = res.x[0]
        mu[i] = res.x[1]
        F[i] = - res.fun
        np.savez(filename, S=S_, phi=phi, mu=mu, F=F)

    print('It took ' + str(round(te - ts, 4)) + 's. For S = ' + str(S))