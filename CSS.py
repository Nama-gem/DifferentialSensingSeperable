import time
from scipy.optimize import minimize, Bounds
from functions import *
from scipy.sparse.linalg import expm

from itertools import combinations

filename = '/users/rey/raka3858/DifferentialSensingSeperable/data/CSS'

try:
    data = np.load(filename + '.npz')
    S_ = data['S']
    phi = data['phi']
    F = data['F']
except:
    S_ = np.append(np.arange(1 / 2, 32, 1 / 2), 2 * np.round(2 ** np.arange(4, 9.25, 0.25)).astype(int))
    phi = np.pi / 4 * np.ones(S_.shape)
    F = np.ones(S_.shape)
    np.savez(filename, S = S_, phi = phi, F = F)


nPhi = int(1E4)

# for i in np.arange(0, len(S_)):
# for i in np.arange(69, len(S_)):
for i in np.arange(0, 72):

    S = S_[i]

    ts = time.time()

    Bx = np.real(expm(- 1j * np.pi / 2 * sy(S).toarray()))
    psi0 = np.zeros(int(2 * S + 1)); psi0[0] = 1
    psi0 = Bx @ psi0

    # method = 'L-BFGS-B'
    method = 'SLSQP'

    tol = 1E-16

    basis = np.kron(Bx, Bx)
    m = np.kron(np.arange(S, - S - 1, - 1), np.ones(int(2 * S + 1))) + np.kron(np.ones(int(2 * S + 1)), np.arange(S, - S - 1, - 1))
    ind_rho = np.concatenate([np.array(list(combinations(np.argwhere(m == m1).ravel(), 2))) for m1 in np.arange(2 * S - 1, - 2 * S, - 1)])

    res = minimize(lambda x: F_prod(x[0], psi0, psi0, basis, ind_rho), phi[i - 1], jac = True,
                   method = method, bounds = Bounds([0 * np.pi], [1 / 2 * np.pi]))

    # res = minimize(lambda x: F_prod(x[0], psi0, psi0, basis, ind_rho), phi[np.max([i + 1, 0])], jac=True,
    #                method=method, bounds=Bounds([0 * np.pi], [1 / 2 * np.pi]))
    # res = minimize(lambda x: F_prod(x[0], psi0, psi0, basis, ind_rho), np.pi / 8, jac = True, method = method, bounds = Bounds([0 * np.pi], [1 / 2 * np.pi]))

    # res = minimize(lambda x: F_prod_fast(x[0], psi0, psi0, nPhi, Bx), phi[np.max([i - 1, 0])], jac = True, method = method)

    te = time.time()

    if (- res.fun > F[i]) or (i == 0):
        print('Improved from ' + str(np.round(dB(F[i]), 4)) + ' to ' + str(np.round(dB(- res.fun), 4)) + ' (dB)')

        if res.x[0] < np.pi / 2 :
            phi[i] = np.mod(res.x[0], np.pi)
        else:
            phi[i] = np.pi - np.mod(res.x[0], np.pi)
        F[i] = - res.fun
        np.savez(filename, S = S_, phi = phi, F = F)

    print('It took ' + str(round(te - ts, 4)) + 's. For S = ' + str(S))
    # break