import numpy as np
import scipy.linalg as sl


hbar = 1
J = 1
beta = 1

delta = 10  # total variation of B


def B(t):  # protocol for B variation
    return B0 + delta/tau*t


def theta(B):  # a useful parameter dependent on B
    return 0.5*np.arctan2(J, B)


def E(B):  # absolute value of the eigen-energy of the system
    return np.sqrt(B**2 + J**2)


def f(r, t):  # array of functions at the right-hand-side of the simultaneous equations
    u, v = r
    fu = 1/(1j*hbar)*(-B(t)*u - J*v)
    fv = 1/(1j*hbar)*(-J*u + B(t)*v)
    return np.array([fu, fv], complex)


def partition(t):  # partition function at time t
    e = E(B(t))
    return np.exp(-beta*e) + np.exp(beta*e)


def rho(final):  # state of the system at the end of the process (time tau)
    e, Z0 = E(B0), partition(0)
    up, vp = final[0, 0], final[0, 1]
    um, vm = final[1, 0], final[1, 1]
    p = np.empty([2, 2], complex)
    p[0, 0] = 1/Z0*(np.exp(-beta*e)*abs(vp)**2 + np.exp(beta*e)*abs(vm)**2)
    p[0, 1] = -1/Z0*(np.exp(-beta*e)*vp*np.conj(up) + np.exp(beta*e)*vm*np.conj(um))
    p[1, 0] = np.conj(p[0, 1])
    p[1, 1] = 1/Z0*(np.exp(-beta*e)*abs(up)**2 + np.exp(beta*e)*abs(um)**2)
    return p


def rho_ad(t):  # state at time t achieved by adiabatic evolution
    e, th, Z0 = E(B0), theta(B(t)), partition(0)
    p = np.empty([2, 2], complex)
    p[0, 0] = 1 / Z0 * (np.exp(-beta * e) * (np.cos(th) ** 2) + np.exp(beta * e) * (np.sin(th)) ** 2)
    p[0, 1] = 1 / Z0 * np.cos(th) * np.sin(th) * (np.exp(-beta * e) - np.exp(beta * e))
    p[1, 0] = np.conj(p[0, 1])
    p[1, 1] = 1 / Z0 * (np.exp(-beta * e) * (np.sin(th) ** 2) + np.exp(beta * e) * (np.cos(th)) ** 2)
    return p


def rho_eq(t):  # equilibrium state at time t
    e, th, Z = E(B(t)), theta(B(t)), partition(t)
    p = np.empty([2, 2], complex)
    p[0, 0] = 1/Z * (np.exp(-beta * e) * (np.cos(th) ** 2) + np.exp(beta * e) * (np.sin(th)) ** 2)
    p[0, 1] = 1/Z * np.cos(th) * np.sin(th) * (np.exp(-beta * e) - np.exp(beta * e))
    p[1, 0] = np.conj(p[0, 1])
    p[1, 1] = 1/Z * (np.exp(-beta * e) * (np.sin(th) ** 2) + np.exp(beta * e) * (np.cos(th)) ** 2)
    return p


def D(p1, p2):  # relative entropy between states p1 and p2
    ln1, ln2 = sl.logm(p1), sl.logm(p2)
    prod1, prod2 = np.matmul(p1, ln1), np.matmul(p1, ln2)
    return np.trace(prod1 - prod2)


def H(t):  # Hamiltonian matrix at time t
    H_t = J*np.ones([2, 2], complex)
    H_t[0, 0], H_t[1, 1] = B(t), -B(t)
    return H_t


def W(final):  # calculates total work done in the process with duration tau
    Ef = np.trace(np.matmul(rho(final), H(tau)))
    return np.real(Ef - E0)


def W_ad(t):  # calculates the work done up to time t if the evolution is adiabatic
    Ef = np.trace(np.matmul(rho_ad(t), H(t)))
    return np.real(Ef - E0)


for B0 in [-10, -5]:
    Bf = B0 + delta

    # list of the process durations that will be investigated
    taupoints = np.logspace(-1, 3, 200)
    n = len(taupoints)
    # lists of total work and final D - each entry corresponds to a process with tau in taupoints:
    Wpoints = np.empty(n, complex)
    Dpoints = np.empty(n, complex)
    D2points = np.empty(n, complex)
    # compute the work for each tau
    for j, tau in enumerate(taupoints):
        a = 0                   # initial time for the simulation
        b = tau                 # time interval for which the simulation will be performed
        N = tau*1000             # number of points to be used in the calculation
        h = (b-a)/N             # spacing between points
        # Array (u, v) for the ('plus') state |+> at t=0
        r_p = np.array([np.sin(theta(B0)), -np.cos(theta(B0))], complex)
        # Array (u, v) for the ('minus') state |-> at t=0
        r_m = np.array([np.cos(theta(B0)), np.sin(theta(B0))], complex)

        # before solving the differential equations, create matrix to save results
        # [[u_p, v_p], [u_m, v_m]]
        results = np.empty([2, 2], complex)

        # solve differential equations by the 4th order Runge-Kutta method
        for i in range(2):
            if i == 0:
                r = r_p
            else:
                r = r_m
            tpoints = np.arange(a, b+h, h)
            xpoints = []
            ypoints = []
            for t in tpoints:
                xpoints.append(r[0])
                ypoints.append(r[1])
                k1 = h*f(r, t)
                k2 = h*f(r+0.5*k1, t+0.5*h)
                k3 = h*f(r+0.5*k2, t+0.5*h)
                k4 = h*f(r+k3, t+h)
                r += (k1+2*k2+2*k3+k4)/6
            results[i] = r.copy()

        # initial energy for computation of the work:
        E0 = np.trace(np.matmul(rho_eq(0), H(0)))
        # compute the work done in the process of duration tau and save it
        Wpoints[j] = W(results)
        # compute the relative entropy and save it
        Dpoints[j] = D(rho(results), rho_eq(tau))
        # compute D[rho(tau)||rho_ad(tau)] and save it
        D2points[j] = D(rho(results), rho_ad(tau))

    # compute W_ad(tau) and D[rho_ad(tau)||rho_eq(tau)] for comparison
    # (they are indepent of tau, so we compute for the last tau only)
    WADpoints = W_ad(tau)*np.ones(n, complex)
    DADpoints = D(rho_ad(tau), rho_eq(tau))*np.ones(n, complex)

    # save everything
    np.savetxt(f'W; B0={B0}', Wpoints)
    np.savetxt(f'W_ad; B0={B0}', WADpoints)
    np.savetxt(f'D; B0={B0}', Dpoints)
    np.savetxt(f'D_ad; B0={B0}', DADpoints)
    np.savetxt(f'D2; B0={B0}', D2points)
