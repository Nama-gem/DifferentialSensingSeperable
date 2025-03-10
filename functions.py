import numpy as np
from scipy import sparse
import sys

def dB(x):
    return 10 * np.log10(x)

# define sparse spin matrices
def op(s, N, K, op):
    if K > N:
        print('site index K should be smaller than the total number of atoms N', file = sys.stderr)
        op = 0
    else:
        op = sparse.kron(sparse.kron(sparse.eye((2 * s + 1) ** (K - 1), format = 'csc'), op), sparse.eye((2 * s + 1) ** (N - K), format = 'csc'), format = 'csc')
    return op

def splus(s, N = 1, K = 1):
    if np.mod(2 * s, 1) != 0:
        print('Choose a valid spin size s', file = sys.stderr)
        splus = 0
    else:
        splus = sparse.csc_matrix((np.array([np.sqrt(s * (s + 1) - M * (M + 1)) for M in np.arange(s - 1, - s - 1, - 1)]),
                                   (np.arange(0, int(2 * s)), np.arange(1, int(2 * s + 1)))),
                                  shape = (int(2 * s + 1), int(2 * s + 1)))
    if N > 1:
        splus = op(s, N, K, splus)
    return splus

def sminus(s, N = 1, K = 1):
    sminus = splus(s).transpose()
    if N > 1:
        sminus = op(s, N, K, sminus)
    return sminus

def sx(s, N = 1, K = 1):
    sx =(splus(s) + sminus(s)) / 2
    if N > 1:
        sx = op(s, N, K, sx)
    return sx

def sy(s, N = 1, K = 1):
    sy = (splus(s) - sminus(s)) / 2j
    if N > 1:
        sy = op(s, N, K, sy)
    return sy

def sz(s, N = 1, K = 1):
    if np.mod(2 * s, 1) != 0:
        print('Choose a valid spin size s', file = sys.stderr)
        sz = 0
    else:
        sz = sparse.csc_matrix((np.arange(s, - s - 1, - 1),
                                   (np.arange(0, int(2 * s + 1)), np.arange(0, int(2 * s + 1)))),
                                  shape = (int(2 * s + 1), int(2 * s + 1)))
    if N > 1:
        sz = op(s, N, K, sz)
    return sz


def oat(mu, s, bx):
    m = np.arange(s, - s - 1, - 1)

    psi0 = np.zeros(int(2 * s + 1))
    psi0[0] = 1
    psi0 = bx @ psi0
    psi_oat = np.exp(- 1j * mu * m ** 2) * psi0

    der_psi_oat = (- 1j * m ** 2) * np.exp(- 1j * mu * m ** 2) * psi0

    if (1 - np.cos(2 * mu) ** (2 * s - 2)) == 0:
        nu = np.pi / 2
        der_nu = 0
    else:
        if (1 - np.cos(2 * mu) ** (2 * s - 2)) == 0:
            print('mu = ' + str(mu))
            print('s = ' + str(s))
        nu = 3 * np.pi / 2 - np.arctan(4 * np.sin(mu) * np.cos(mu) ** (2 * s - 2) / (1 - np.cos(2 * mu) ** (2 * s - 2))) / 2
        der_nu = (np.cos(mu) ** (1 + 2 * s) * np.cos(2 * mu) * (np.cos(2 * mu) ** (2 * s) * (
                    - 5 + 6 * s + (6 - 4 * s) * np.cos(2 * mu) + (3 - 2 * s) * np.cos(4 * mu)) - 4 * np.cos(2 * mu) ** 3 *
                                                           (np.cos(mu) ** 2 - 2 * (- 1 + s) * np.sin(mu) ** 2))) / (
                          2 * np.cos(mu) ** 4 * (np.cos(2 * mu) ** 2 - np.cos(2 * mu) ** (2 * s)) ** 2 +
                          32 * np.cos(mu) ** (4 * s) * np.cos(2 * mu) ** 4 * np.sin(mu) ** 2)

    der_psi_oat = bx @ ((- 1j * m) * der_nu * np.exp(- 1j * nu * m) * (bx.T @ psi_oat)) + bx @ (
                np.exp(- 1j * nu * m) * (bx.T @ der_psi_oat))
    psi_oat = bx @ (np.exp(- 1j * nu * m) * (bx.T @ psi_oat))

    return psi_oat, der_psi_oat


# ind_rho are the indices where Jz rho = rho Jz
def p_prod_der(phi, psi_a, psi_b, basis, ind_rho):
    s = (len(psi_a[0]) - 1) / 2
    d = int(2 * s + 1) ** 2
    m = np.arange(s, - s - 1, - 1)

    psi_a_phi = np.exp(- 1j * phi * m) * psi_a[0]
    psi_b_phi = np.exp(+ 1j * phi * m) * psi_b[0]
    der_psi_a_phi = np.exp(- 1j * phi * m) * psi_a[1]
    der_psi_b_phi = np.exp(+ 1j * phi * m) * psi_b[1]
    derphi_psi_a_phi = (- 1j * m) * np.exp(- 1j * phi * m) * psi_a[0]
    derphi_psi_b_phi = (+ 1j * m) * np.exp(+ 1j * phi * m) * psi_b[0]
    derphi_der_psi_a_phi = (- 1j * m) * np.exp(- 1j * phi * m) * psi_a[1]
    derphi_der_psi_b_phi = (+ 1j * m) * np.exp(+ 1j * phi * m) * psi_b[1]
    derphiderphi_psi_a_phi = (- 1j * m) ** 2 * np.exp(- 1j * phi * m) * psi_a[0]
    derphiderphi_psi_b_phi = (+ 1j * m) ** 2 * np.exp(+ 1j * phi * m) * psi_b[0]

    psi = np.kron(psi_a_phi, psi_b_phi)
    derphi_psi = np.kron(derphi_psi_a_phi, psi_b_phi) + np.kron(psi_a_phi, derphi_psi_b_phi)
    derphiderphi_psi = np.kron(derphiderphi_psi_a_phi, psi_b_phi) + np.kron(psi_a_phi, derphiderphi_psi_b_phi) + 2 * np.kron(derphi_psi_a_phi, derphi_psi_b_phi)
    der_psi = np.kron(der_psi_a_phi, psi_b_phi) + np.kron(psi_a_phi, der_psi_b_phi)
    derphi_der_psi = np.kron(derphi_der_psi_a_phi, psi_b_phi) + np.kron(derphi_psi_a_phi, der_psi_b_phi) + np.kron(der_psi_a_phi, derphi_psi_b_phi) + np.kron(psi_a_phi, derphi_der_psi_b_phi)

    rho = sparse.csc_matrix((psi[ind_rho[:, 0]] * psi[ind_rho[:, 1]].conj(),
                           (ind_rho[:, 0], ind_rho[:, 1])), shape=(d, d))
    rho = rho + rho.conj().T + sparse.diags(np.real(psi.conj() * psi), 0, format='csc')

    derphi_rho = sparse.csc_matrix((derphi_psi[ind_rho[:, 0]] * psi[ind_rho[:, 1]].conj() + psi[ind_rho[:, 0]] * derphi_psi[ind_rho[:, 1]].conj(),
                              (ind_rho[:, 0], ind_rho[:, 1])), shape=(d, d))
    derphi_rho = derphi_rho + derphi_rho.conj().T + sparse.diags(np.real(derphi_psi.conj() * psi + psi.conj() * derphi_psi), 0, format='csc')

    der_rho = sparse.csc_matrix((der_psi[ind_rho[:, 0]] * psi[ind_rho[:, 1]].conj() + psi[ind_rho[:, 0]] * der_psi[ind_rho[:, 1]].conj(),
                             (ind_rho[:, 0], ind_rho[:, 1])), shape=(d, d))
    der_rho = der_rho + der_rho.conj().T + sparse.diags(np.real(der_psi.conj() * psi + psi.conj() * der_psi), 0, format='csc')

    derphiderphi_rho = sparse.csc_matrix((derphiderphi_psi[ind_rho[:, 0]] * psi[ind_rho[:, 1]].conj() + psi[ind_rho[:, 0]] * derphiderphi_psi[
        ind_rho[:, 1]].conj() + 2 * derphi_psi[ind_rho[:, 0]] * derphi_psi[ind_rho[:, 1]].conj(),
                                (ind_rho[:, 0], ind_rho[:, 1])), shape=(d, d))
    derphiderphi_rho = derphiderphi_rho + derphiderphi_rho.conj().T + sparse.diags(
        np.real(derphiderphi_psi.conj() * psi + psi.conj() * derphiderphi_psi + 2 * derphi_psi.conj() * derphi_psi), 0, format='csc')

    derphi_der_rho = sparse.csc_matrix((derphi_der_psi[ind_rho[:, 0]] * psi[ind_rho[:, 1]].conj() + psi[ind_rho[:, 0]] * derphi_der_psi[
        ind_rho[:, 1]].conj() + derphi_psi[ind_rho[:, 0]] * der_psi[ind_rho[:, 1]].conj() + der_psi[ind_rho[:, 0]] * derphi_psi[ind_rho[:, 1]].conj(),
                                (ind_rho[:, 0], ind_rho[:, 1])), shape=(d, d))
    derphi_der_rho = derphi_der_rho + derphi_der_rho.conj().T + sparse.diags(
        np.real(derphi_der_psi.conj() * psi + psi.conj() * derphi_der_psi + derphi_psi.conj() * der_psi + der_psi.conj() * derphi_psi), 0, format='csc')

    p = np.real(np.sum((basis @ rho) * basis.conj(), axis=1))
    der_p = np.real(np.sum((basis @ der_rho) * basis.conj(), axis=1))
    derphi_p = np.real(np.sum((basis @ derphi_rho) * basis.conj(), axis=1))
    derphi_der_p = np.real(np.sum((basis @ derphi_der_rho) * basis.conj(), axis=1))
    derphiderphi_p = np.real(np.sum((basis @ derphiderphi_rho) * basis.conj(), axis=1))

    return p, der_p, derphi_p, derphi_der_p, derphiderphi_p


def F_prod_der(phi, psi_a, psi_b, basis, ind_rho):
    p, der_p, derphi_p, derphi_der_p, derphiderphi_p = p_prod_der(phi, psi_a, psi_b, basis, ind_rho)

    derphiderphi_p = derphiderphi_p[p > 0];
    derphi_der_p = derphi_der_p[p > 0];
    derphi_p = derphi_p[p > 0];
    der_p = der_p[p > 0];
    p = p[p > 0]

    F = np.sum(derphi_p ** 2 / p, axis=0)

    derphi_F = np.sum(2 * derphi_p * derphiderphi_p / p - derphi_p ** 3 / p ** 2)

    der_F = np.sum(2 * derphi_p * derphi_der_p / p - derphi_p ** 2 * der_p / p ** 2)

    J_F = np.array([derphi_F, der_F])

    return - F, - J_F


def F_prod_fast_der(phi, psi_a, psi_b, nPhi, bx):
    s = (len(psi_a[0]) - 1) / 2
    d = int(2 * s + 1)
    m = np.arange(s, - s - 1, - 1)

    psi_a_phi = np.exp(- 1j * phi * m) * psi_a[0];
    psi_b_phi = np.exp(+ 1j * phi * m) * psi_b[0];
    derphi_psi_a_phi = (- 1j * m) * psi_a_phi;
    derphi_psi_b_phi = (+ 1j * m) * psi_b_phi;
    derphiderphi_psi_a_phi = (- 1j * m) * derphi_psi_a_phi;
    derphiderphi_psi_b_phi = (+ 1j * m) ** 2 * psi_b_phi;
    der_psi_a_phi = np.exp(- 1j * phi * m) * psi_a[1];
    der_psi_b_phi = np.exp(+ 1j * phi * m) * psi_b[1];
    derphi_der_psi_a_phi = (- 1j * m) * der_psi_a_phi;
    derphi_der_psi_b_phi = (+ 1j * m) * der_psi_b_phi;

    tmp = np.array([bx.reshape(1, d, d) @ (np.exp(+ 1j * Phi * m.reshape((1, d))) *
                                           np.array(
                                               [psi_a_phi, psi_b_phi, derphi_psi_a_phi, derphi_psi_b_phi, derphiderphi_psi_a_phi, derphiderphi_psi_b_phi, der_psi_a_phi, der_psi_b_phi, derphi_der_psi_a_phi,
                                                derphi_der_psi_b_phi])).reshape((10, d, 1)) for Phi in
                    np.pi * np.arange(- 1, 1, 2 / nPhi)])

    p = np.real(
        np.sum([np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 1].conj()) for i in range(nPhi)], axis=0) / nPhi)

    # note that derphi_psi = kron(derphi_psi_A, psi_B) + kron(psi_A, derphi_psi_B)
    derphi_p = np.sum([2 * np.real(np.kron(tmp[i, 0] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
                               np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 3].conj()))
                   for i in range(nPhi)], axis=0) / nPhi

    derphiderphi_p = np.sum([2 * np.real(np.kron(tmp[i, 0] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
                                 np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 3].conj()))
                     for i in range(nPhi)], axis=0) / nPhi

    der_p = np.sum([2 * np.real(np.kron(tmp[i, 0] * tmp[i, 6].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
                              np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 7].conj()))
                  for i in range(nPhi)], axis=0) / nPhi

    derphiderphi_p = np.sum([np.real(
        2 * np.kron(tmp[i, 0] * tmp[i, 4].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 5].conj()) +
        2 * np.kron(tmp[i, 2] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 3] * tmp[i, 3].conj()) +
        8 * np.kron(tmp[i, 0] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 3].conj()))
        for i in range(nPhi)], axis=0) / nPhi

    derphi_der_p = np.sum([np.real(
        2 * np.kron(tmp[i, 8] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 6] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 6] * tmp[i, 0].conj(), tmp[i, 3] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 6] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 3].conj()) +

        2 * np.kron(tmp[i, 2] * tmp[i, 0].conj(), tmp[i, 7] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 0] * tmp[i, 2].conj(), tmp[i, 7] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 9] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 7] * tmp[i, 3].conj())
    )
        for i in range(nPhi)], axis=0) / nPhi

    del tmp

    F = np.sum(derphi_p ** 2 / p, axis=0)

    derphi_F = np.sum(2 * derphi_p * derphiderphi_p / p - derphi_p ** 3 / p ** 2)

    der_F = np.sum(2 * derphi_p * derphi_der_p / p - derphi_p ** 2 * der_p / p ** 2)

    del p, der_p, derphi_p, derphi_der_p, derphiderphi_p

    J_F = np.array([derphi_F, der_F])

    return - F, - J_F


# ind_rho are the indices where Jz rho = rho Jz
def p_prod(phi, psi_a, psi_b, basis, ind_rho):
    s = (len(psi_a) - 1) / 2
    d = int(2 * s + 1) ** 2
    m = np.arange(s, - s - 1, - 1)

    psi_a_phi = np.exp(- 1j * phi * m) * psi_a
    psi_b_phi = np.exp(+ 1j * phi * m) * psi_b
    derphi_psi_a_phi = (- 1j * m) * np.exp(- 1j * phi * m) * psi_a
    derphi_psi_b_phi = (+ 1j * m) * np.exp(+ 1j * phi * m) * psi_b
    derphiderphi_psi_a_phi = (- 1j * m) ** 2 * np.exp(- 1j * phi * m) * psi_a
    derphiderphi_psi_b_phi = (+ 1j * m) ** 2 * np.exp(+ 1j * phi * m) * psi_b

    psi = np.kron(psi_a_phi, psi_b_phi);
    derphi_psi = np.kron(derphi_psi_a_phi, psi_b_phi) + np.kron(psi_a_phi, derphi_psi_b_phi)
    derphiderphi_psi = np.kron(derphiderphi_psi_a_phi, psi_b_phi) + np.kron(psi_a_phi, derphiderphi_psi_b_phi) + 2 * np.kron(derphi_psi_a_phi, derphi_psi_b_phi)

    rho = sparse.csc_matrix((psi[ind_rho[:, 0]] * psi[ind_rho[:, 1]].conj(),
                           (ind_rho[:, 0], ind_rho[:, 1])), shape=(d, d))
    rho = rho + rho.conj().T + sparse.diags(np.real(psi.conj() * psi), 0, format='csc')

    derphi_rho = sparse.csc_matrix((derphi_psi[ind_rho[:, 0]] * psi[ind_rho[:, 1]].conj() + psi[ind_rho[:, 0]] * derphi_psi[ind_rho[:, 1]].conj(),
                              (ind_rho[:, 0], ind_rho[:, 1])), shape=(d, d))
    derphi_rho = derphi_rho + derphi_rho.conj().T + sparse.diags(np.real(derphi_psi.conj() * psi + psi.conj() * derphi_psi), 0, format='csc')

    derphiderphi_rho = sparse.csc_matrix((derphiderphi_psi[ind_rho[:, 0]] * psi[ind_rho[:, 1]].conj() + psi[ind_rho[:, 0]] * derphiderphi_psi[
        ind_rho[:, 1]].conj() + 2 * derphi_psi[ind_rho[:, 0]] * derphi_psi[ind_rho[:, 1]].conj(),
                                (ind_rho[:, 0], ind_rho[:, 1])), shape=(d, d))
    derphiderphi_rho = derphiderphi_rho + derphiderphi_rho.conj().T + sparse.diags(
        np.real(derphiderphi_psi.conj() * psi + psi.conj() * derphiderphi_psi + 2 * derphi_psi.conj() * derphi_psi), 0, format='csc')

    p = np.real(np.sum((basis @ rho) * basis.conj(), axis=1))
    derphi_p = np.real(np.sum((basis @ derphi_rho) * basis.conj(), axis=1))
    derphiderphi_p = np.real(np.sum((basis @ derphiderphi_rho) * basis.conj(), axis=1))

    return p, derphi_p, derphiderphi_p


def F_prod(phi, psi_a, psi_b, basis, ind_rho):
    p, derphi_p, derphiderphi_p = p_prod(phi, psi_a, psi_b, basis, ind_rho)

    derphiderphi_p = derphiderphi_p[p > 0];
    derphi_p = derphi_p[p > 0];
    p = p[p > 0]

    F = np.sum(derphi_p ** 2 / p, axis=0)

    derphi_F = np.sum(2 * derphi_p * derphiderphi_p / p - derphi_p ** 3 / p ** 2)

    return - F, - derphi_F


def F_prod_fast(phi, psi_a, psi_b, nPhi, bx):
    s = (len(psi_a) - 1) / 2
    d = int(2 * s + 1)
    m = np.arange(s, - s - 1, - 1)

    psi_a_phi = np.exp(- 1j * phi * m) * psi_a
    psi_b_phi = np.exp(+ 1j * phi * m) * psi_b
    derphi_psi_a_phi = (- 1j * m) * psi_a_phi
    derphi_psi_b_phi = (+ 1j * m) * psi_b_phi
    derphiderphi_psi_a_phi = (- 1j * m) * derphi_psi_a_phi
    derphiderphi_psi_b_phi = (+ 1j * m) ** 2 * psi_b_phi

    tmp = np.array([bx.reshape(1, d, d) @ (np.exp(+ 1j * Phi * m.reshape((1, d))) *
                                           np.array([psi_a_phi, psi_b_phi, derphi_psi_a_phi, derphi_psi_b_phi, derphiderphi_psi_a_phi, derphiderphi_psi_b_phi])).reshape((6, d, 1))
                    for Phi in np.pi * np.arange(- 1, 1, 2 / nPhi)])

    p = np.real(
        np.sum([np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 1].conj()) for i in range(nPhi)], axis=0) / nPhi)

    # note that derphi_psi = kron(derphi_psi_A, psi_B) + kron(psi_A, derphi_psi_B)
    derphi_p = np.sum([2 * np.real(np.kron(tmp[i, 0] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
                               np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 3].conj()))
                   for i in range(nPhi)], axis=0) / nPhi

    derphiderphi_p = np.sum([2 * np.real(np.kron(tmp[i, 0] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
                                 np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 3].conj()))
                     for i in range(nPhi)], axis=0) / nPhi

    derphiderphi_p = np.sum([np.real(
        2 * np.kron(tmp[i, 0] * tmp[i, 4].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 1] * tmp[i, 5].conj()) +
        2 * np.kron(tmp[i, 2] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 1].conj()) +
        2 * np.kron(tmp[i, 0] * tmp[i, 0].conj(), tmp[i, 3] * tmp[i, 3].conj()) +
        8 * np.kron(tmp[i, 0] * tmp[i, 2].conj(), tmp[i, 1] * tmp[i, 3].conj()))
        for i in range(nPhi)], axis=0) / nPhi

    del tmp

    F = np.sum(derphi_p ** 2 / p, axis=0)

    derphi_F = np.sum(2 * derphi_p * derphiderphi_p / p - derphi_p ** 3 / p ** 2)

    del p, derphi_p, derphiderphi_p

    return - F, - derphi_F