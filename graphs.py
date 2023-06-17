import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook', 'grid'])


for B0 in [-10, -5]:
    # list of the process durations that will be investigated
    taupoints = np.logspace(-1, 3, 200)
    # reading the previously calculated values for W
    Wpoints = np.loadtxt(f'W; B0={B0}', complex)
    WADpoints = np.loadtxt(f'W_ad; B0={B0}', complex)

    Dpoints = np.loadtxt(f'D; B0={B0}', complex)
    DADpoints = np.loadtxt(f'D_ad; B0={B0}', complex)
    D2points = np.loadtxt(f'D2; B0={B0}', complex)

    for i, tau in enumerate(taupoints):
        if tau > 300:
            break

    # plot W in linear scale
    fig, ax = plt.subplots()
    ax.plot(taupoints, np.real(Wpoints), 'r.', label=r'$W_\tau(\tau)$')
    ax.plot(taupoints, np.real(WADpoints), 'k--', label=r'$W_{ad}(\tau)$')
    ax.legend(fancybox=False, edgecolor='black')
    ax.set_xlabel(r'$\tau\cdot J$')
    ax.set_ylabel('$W \cdot J^{-1}$')
    ax.set_title(r'$\tau$-dependence of the total work $W_\tau(\tau)$')
    ax.set_xlim(0, 100)
    ax.set_box_aspect(1)
    plt.savefig(f'F:\IC\Landau-Zener\B0 = {B0}/W(tau),B0={B0}.png', bbox_inches='tight')

    # fit for log scale
    fit = np.polyfit(np.log(taupoints[i:]), np.log(np.real(Wpoints[i:]-WADpoints[i:])), deg=1, cov=True)
    print(f'For B0={B0}:')
    print(fit)
    W_fit = taupoints**fit[0][0] * np.exp(fit[0][1])

    # plot Wex in log scale
    Wex = np.real(Wpoints-WADpoints)
    fig2, ax2 = plt.subplots()
    ax2.loglog(taupoints, Wex, 'r.', label=r'$[W_\tau(\tau)-W_{ad}(\tau)]\cdot J^{-1}$')
    ax2.loglog(taupoints, W_fit, 'k--', label=r'Fit for $\tau \cdot J \gg 1$')
    ax2.legend(fancybox=False, edgecolor='black')
    ax2.set_xlabel(r'$\tau\cdot J$')
    ax2.set_ylabel('$W \cdot J^{-1}$')
    ax2.set_title(r'$\tau$-dependence of the excess work')
    ax2.set_xlim(1e-1, 1000)
    ax2.set_ylim(0.8*min(Wex), 1.2*max(Wex))
    ax2.set_box_aspect(1)
    plt.savefig(f'F:\IC\Landau-Zener\B0 = {B0}/log(W)_log(tau),B0={B0}.png', bbox_inches='tight')

    # plot D in linear scale
    fig3, ax3 = plt.subplots()
    ax3.plot(taupoints, np.real(Dpoints), 'r.', label=r'$D[\rho_\tau (\tau) || \rho_{eq} (\tau)]$')
    ax3.plot(taupoints, np.real(DADpoints), 'k--', label=r'$D[\rho_{ad} (\tau) || \rho_{eq} (\tau)]$')
    ax3.legend(fancybox=False, edgecolor='black')
    ax3.set_xlabel(r'$\tau\cdot J$')
    ax3.set_ylabel('Relative entropy')
    ax3.set_title(r'$\tau$-dependence of the relative entropy $D$')
    ax3.set_xlim(0, 100)
    ax3.set_box_aspect(1)
    plt.savefig(f'F:\IC\Landau-Zener\B0 = {B0}/D(tau),B0={B0}.png', bbox_inches='tight')

    # plot D in log scale
    fig4, ax4 = plt.subplots()
    ax4.loglog(taupoints, np.real(Dpoints), 'r.', label=r'$D[\rho_\tau (\tau) || \rho_{eq} (\tau)]$')
    ax4.legend(fancybox=False, edgecolor='black')
    ax4.set_xlabel(r'$\tau\cdot J$')
    ax4.set_ylabel('Relative entropy')
    ax4.yaxis.grid(True, which='minor')
    ax4.set_xlim(0.1, 1000)
    ax4.set_title(r'$\tau$-dependence of the relative entropy')
    ax4.set_box_aspect(1)
    plt.savefig(f'F:\IC\Landau-Zener\B0 = {B0}/log(D)_log(tau),B0={B0}.png', bbox_inches='tight')

    # plot Dex in log scale together with Wex
    Dex = np.real(Dpoints-DADpoints)
    fig5, ax5 = plt.subplots()
    ax5.loglog(taupoints[0::2], Wex[0::2], 'r.', label=r'$W_{ex}(\tau) \cdot J^{-1}$')
    ax5.loglog(taupoints[1::2], Dex[1::2], 'k.', label=r'$D_{ex}(\tau)$')
    ax5.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax5.set_xlabel(r'$\tau\cdot J$')
    ax5.set_ylabel('$W \cdot J^{-1} \ $ or $\ \ D$')
    ax5.set_title(r'Comparison between $W_{ex}$ and $D_{ex}$')
    ax5.set_xlim(1e-1, 1000)
    ax5.set_box_aspect(1)
    plt.savefig(f'F:\IC\Landau-Zener\B0 = {B0}/Wex vs Dex, B0={B0}.png', bbox_inches='tight')

    # plot D in log scale with Wex
    fig6, ax6 = plt.subplots()
    ax6.loglog(taupoints[0::2], Wex[0::2], 'r.', label=r'$[W_\tau(\tau)-W_{ad}(\tau)]\cdot J^{-1}$')
    ax6.loglog(taupoints[1::2], np.real(Dpoints[1::2]), 'k.', label=r'$D[\rho_\tau (\tau) || \rho_{eq} (\tau)]$')
    ax6.legend(fancybox=False, edgecolor='black')
    ax6.set_xlabel(r'$\tau\cdot J$')
    ax6.set_ylabel('$W \cdot J^{-1} \ $ or $\ \ D$')
    ax6.set_title(r'Comparison between $W_{ex}$ and $D$')
    ax6.set_xlim(1e-1, 1000)
    ax6.set_box_aspect(1)
    plt.savefig(f'F:\IC\Landau-Zener\B0 = {B0}/Wex vs D, B0={B0}.png', bbox_inches='tight')

    # plot D2 in log scale together with Wex
    fig7, ax7 = plt.subplots()
    ax7.loglog(taupoints[0::2], Wex[0::2], 'r.', label=r'$W_{\tau,ex}(\tau) \cdot J^{-1}$')
    ax7.loglog(taupoints[1::2], Dex[1::2], 'k.', label=r'$D[\rho_\tau(\tau)||\rho_{ad}(\tau)]$')
    ax7.legend(fancybox=False, edgecolor='black', loc='lower left')
    ax7.set_xlabel(r'$\tau\cdot J$')
    ax7.set_ylabel('$W \cdot J^{-1} \ $ or $\ \ D$')
    ax7.set_title(r'Comparison between $W_{\tau,ex}(\tau)$ and $D[\rho_\tau(\tau)||\rho_{ad}(\tau)]$')
    ax7.set_xlim(1e-1, 1000)
    ax7.set_box_aspect(1)
    plt.savefig(f'F:\IC\Landau-Zener\B0 = {B0}/Wex vs D(rho,rho_ad), B0={B0}.png', bbox_inches='tight')
