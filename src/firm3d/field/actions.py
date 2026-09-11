import os

import numpy as np
import matplotlib.pyplot as plt

from firm3d.field.boozermagneticfield import (
    BoozerAnalytic
)

from firm3d.field.tracing_helpers import (
    initialize_position_uniform_s_chi_grid,
)

from firm3d.field.action_helpers import (
    check_expression_conditionals,
    check_expression_conditionals_finite_mu
)

from scipy.interpolate import CubicSpline
from scipy import optimize

__all__ = [
    "JchiAction"
]

class JchiAction(object):
    r"""A class that computes the action $\J_\chi$.

    This class computes the action $J_\chi$ for a near-axis expansion magnetic
    field (Landreman & Sengupta, JPP 2018). The action is computed as an
    expansion in $\rho_\star$ at both 0th order and 1st order. The action is
    calculated by integrating the canonical momenta derived from the
    Littlejohn guiding center lagrangian.
    The momenta are dimensionless according to a length scale, time scale, and 
    mass scale. The mass scale should be the mass of an alpha particle, and the
    length scale should be the major radius of the device. The time scale is 
    determined by relating the mass and length scales to the birth energy of an 
    alpha particle.
    All calculations are done with the assuption that the particle is purely
    passing ($\mu = 0$).
    """
    def __init__(self, Ekin, mass, q, outdir=None, subdir=None):
        """
        ComputingJChi class is initialized with 3 physical parameters that 
        determine the type of particle and its energy. 

        The other two parameters determine the output directory for plots.

        Args:
            Ekin (float): The kinetic energy of the particles species
            mass (float): The mass of the species, and is also the normalizing
                mass scale.
            q (float): The charge of the species we are calculating the action
                for.
            outdir (string): The subdirectory created in the parent directory
                where this class is run. Plots are placed into this folder.
            subdir (string): Another subdirectory that may be created inside
                of 'outdir'. Useful if computing the action with varying 
                parameters.

        Returns:
            None
        """
        self.Ekin = Ekin
        self.mass = mass
        self.q = q

        self.vpar0 = np.sqrt(2 * self.Ekin / self.mass)
        
        self.outdir = outdir
        self.subdir = subdir
        if self.outdir:
            os.makedirs(self.outdir, exist_ok=True)
        if self.subdir:
            os.makedirs(os.path.join(self.outdir, self.subdir), exist_ok=True)

    def initField(self, B0, R, a, iota0, helicity_M, helicity_N, helicity_Mp, helicity_Np):
        r"""
        Initializes the analytic field parameters for the near-axis form of the
        magnetic field. 

        The analytic form of the magnetic field is
            |B| = B_0 (1 - \sqrt{2\psi/\bar{B}} \bar{\eta} \cos(\chi))

        \psi is estimated fromm B0 as \psi = \pi a^2 B0
        \bar{B} is set to be B0

        Note that the combination of the above makes prefactor in front of 
        \cos(\chi) independent of B0.

        The mass and length normalization scales are set to be mass and major 
        radius at this point, and the time normalization scale based off the 
        energy as well. 

        Args:
            B0 (float): Magnitude of the magnetic field
            R (float): Major radius of the device
            a (float): Minor radius of the device
            iota0 (float): Rotational transform of the field
            helicity_M (int): Poloidal helicity of the magnetic field
            helicity_N (int): Toroidal helicity of the magnetic field
            helicity_Mp (int): Poloidal helicity of the mapping coordinate eta
            helicity_Np (int): Toroidal helicity of the mapping coordinate eta

        Returns:
            None
        """
        self.B0 = B0
        self.R = R
        self.a = a
        self.iota0 = iota0
        self.helicity_M = helicity_M
        self.helicity_N = helicity_N
        self.helicity_Mp = helicity_Mp
        self.helicity_Np = helicity_Np

        if (self.helicity_Mp * self.helicity_N) == (self.helicity_Np * self.helicity_M):
            raise ValueError(
                "Chosen helicities (N, M, N', M') do not create a well "
                "defined Jacobian."
            )
        
        self.jac = self.helicity_Mp*self.helicity_N - self.helicity_M*self.helicity_Np

        self.G0 = self.B0*self.R
        self.etabar = 1/self.R
        self.psi_b = (np.pi * a**2 * self.B0) / (2*np.pi) 

        self.m_norm = self.mass
        self.L_norm = self.R
        self.T_norm = np.sqrt((self.m_norm * self.L_norm**2) / self.Ekin)
        self.Hnorm = self.m_norm * self.L_norm**2 / self.T_norm**2

        self.field = BoozerAnalytic(
            etabar = self.etabar,
            B0 = self.B0,
            N = self.helicity_N,
            G0 = self.G0,
            psi0 = self.psi_b,
            iota0 = self.iota0,
            Bbar = self.B0
        )     

    def initGrids(self, ns, nchi):
        r"""
        Initializes the grids over which plots of the canonical momenta and
        modB are generated. 

        A set of points evenly spaced in $(s, \chi)$ is generated, and used to 
        create 2D arrays in s and \chi used for plotting.

        Args:
            ns (int): The resolution in s
            nchi (int): The resolution in $\chi$

        Returns:
            None
        """
        self.ns = ns
        self.nchi = nchi
        self.points = initialize_position_uniform_s_chi_grid(
            self.field, self.ns, self.nchi, self.helicity_M, self.helicity_N
        )

        s_arr = self.points[:, 0]
        theta_arr = self.points[:, 1]
        zeta_arr = self.points[:, 2]

        self.s_mat = s_arr.reshape(self.ns, self.nchi)
        theta_mat = theta_arr.reshape(self.ns, self.nchi)
        zeta_mat = zeta_arr.reshape(self.ns, self.nchi)

        chi_arr = self.helicity_M*theta_arr - self.helicity_N*zeta_arr
        chi_arr = chi_arr % (2*np.pi)
        chi_mat = self.helicity_M*theta_mat - self.helicity_N*zeta_mat
        self.chi_mat = chi_mat % (2*np.pi)

        eta_mat = self.helicity_Mp*theta_mat - self.helicity_Np*zeta_mat
        self.eta_mat = eta_mat % (2*np.pi)
        
    def compute_normalized_momenta(self, points, vpar, terms=False):
        """
        Computes the normalized canonical momenta term by term.

        args:
            points (np.array(npoints, 3)): An numpy array (or list) of dimension
                (npoints, 3) representing s, theta, zeta coordinates
            vpar (np.array(npoints)): A numpy array (or list) of dimension 
                (npoints) representing the vpar value for the different
                locations corresponding to points.
            terms (bool): A flag which determines whether the individual terms
                making up the momenta or the full momenta are returned.
                if terms=False, the total $P_\chi$ and $P_\eta$ are returned.
                if terms=True, the individual terms making up $P_\chi$ and 
                $P_\eta$ are returned.
        
        returns:
            Normalized canonical momenta 
        """
        # Get field relevant values
        self.field.set_points(points)
        modB = self.field.modB()[:, 0]
        G = self.field.G()[:, 0]
        psi = self.field.psi0 * np.array(points)[:, 0]
        iota0 = self.field.iota0
        rhopar = self.mass * vpar / (self.q * modB)

        # Compute normalizations for G and psi
        G_norm = (self.m_norm * self.L_norm) / (self.q* self.T_norm)
        psi_norm = (self.m_norm * self.L_norm**2) / (self.q * self.T_norm)

        # Compute normalized quantities
        rhobar = rhopar / self.L_norm
        Gbar = G / G_norm
        psibar = psi / psi_norm

        # Compute rho_\parallel and \psi terms for Pchi and Peta
        denom = self.jac
        pchi_term1 =  -self.helicity_Mp * Gbar * rhobar / denom
        pchi_term2 = psibar * (self.helicity_Mp*iota0 - self.helicity_Np) / denom
        peta_term1 = self.helicity_M * Gbar * rhobar / denom
        peta_term2 = psibar * (self.helicity_N - self.helicity_M*iota0) / denom

        if terms:
            return pchi_term1, pchi_term2, peta_term1, peta_term2
        else:
            pchi = pchi_term1 + pchi_term2
            peta = peta_term1 + peta_term2
            return pchi, peta

    def getJChiNumeric(self, peta_target, nchi):
        """
        Computes J_\chi by numerical integration over a phase space trajectory
        of a purely passing (\mu=0) particle.

        The integration is performed over a phase space trajectory, which 
        requires arrays in the canonical coordinate \chi and the canonical
        momentum P_\chi.

        An array over a cycle in \chi from 0 to 2\Pi is generated. For each 
        \chi value, the s coordinate that corresponds to 'peta_target' is 
        computed. 

        The s coordinate are saved into a list. 

        The s and chi lists are used to create a list of points (s, theta, zeta)
        over which P_\chi is computed.

        The resulting P_\chi array is determined at constant P_\eta (as well as
        energy and \eta).

        Finally, the array is integrated over the cycle to compute the numerical
        action.

        Args:
            peta_target (float): The target peta value used in the root solve.
            nchi (int): Resolution of phase space trajectory used in the
                integration.

        Return:
            Jchi_numeric (float): Numerical estimate of J_\chi.
        """
        # Create evenly spaced points in chi, as well as s array to store
        # s values
        chis = np.linspace(0, 2*np.pi, nchi, endpoint=False)
        ss = np.zeros(len(chis))
        # Eta is needed to convert from chi to theta & zeta, and must be
        # constant in the integration
        eta_val = 0
        etas = np.full_like(chis, fill_value=eta_val)

        # Iterate ove rchi values
        for i, chi_val in enumerate(chis):
            # Define a function to pass into scipy.optimize.root_scalar
            # that takes in one argument s.
            # This could probably be simplified by defining it earlier and 
            # using a lambda function so that s can be passed in, in addition to
            # chi_val and eta_val, but instead we redefine the function every
            # time.
            def root_peta(s):
                theta = (1 / self.jac) * (
                    -self.helicity_Np * chi_val + self.helicity_N * eta_val
                )
                zeta = (1 / self.jac) * (
                    -self.helicity_Mp * chi_val + self.helicity_M * eta_val
                )
                # Save coordinates as a point
                point = np.array([s, theta, zeta])

                # Extracts Peta and then extracts value from array
                peta = self.compute_normalized_momenta(
                    [point], 
                    self.vpar0, 
                    terms=False
                )[1][0] 
                # The root of this function is when peta = peta_target
                return peta_target - peta
            
            x = optimize.root_scalar(
                root_peta,
                x0 = 0.5, 
            )
            # If successful, save the root into an array
            if x.converged and x.root > 0:
                s_target = x.root
                ss[i] = s_target
            else:
                print(x.converged)
                print(x.root)
                raise Exception(f"Root solve failed for Peta = {peta_target}!")
        ss = np.array(ss)
        # Useful to later visualize how the deviation of the drift surfaces to 
        # the flux surfaces changes.
        self.sdiff = np.max(ss) - np.min(ss) 
        # Built points out of s, theta, zeta arr
        thetas = (1 / self.jac) * (
            -self.helicity_Np * chis + self.helicity_N * etas
        )
        zeta = (1 / self.jac) * (
            -self.helicity_Mp * chis + self.helicity_M * etas
        )
        points_constPeta = np.vstack((ss, thetas, zeta)).T
        points_constPeta = np.ascontiguousarray(points_constPeta, dtype=np.float64)
        vpar = np.full(shape=nchi, fill_value = self.vpar0)
        # Compute Pchi at constant Peta
        pchi_constPeta = self.compute_normalized_momenta(
            points_constPeta, 
            vpar, 
            terms=False
        )[0]
        # Perform integration to compute Jchi
        Jchi_numeric =  (np.sum(pchi_constPeta) * (chis[1]-chis[0])) / (2*np.pi)
        # Downsample the number of points used in plotting later
        self.ss_plot = ss[::nchi//10]
        self.chis_plot = chis[::nchi//10]

        return Jchi_numeric

    def getJchi(self, peta, nchi):
        """
        Call functions to compute numerical and analytic functions, and save 
        them as member variables

        Args:
            peta (float): Peta value used to compute the actions
            nchi (int): The resolution in chi for the numerical action integral
        
        Returns:
            None
        """
        self.JChi_numeric = self.getJChiNumeric(peta, nchi)
        self.JChi0_analytic = self.getJChiAnalytic0thOrder(peta)
        self.JChi1_analytic = self.getJChiAnalytic1stOrder(peta)
        
    def getJChiAnalytic0thOrder(self, peta):
        r"""
        Computes the 0th order analytic action derived in the NAE, \rho_\star
        expansion of the canonical momenta.

        Args:
            peta (float): 0th order Peta value used in calculation

        Returns:
            Jchi0_analytic (float): 0th order J_\chi
        """
        Jchi0_analytic = (
            peta * (
                    (
                        (self.helicity_Mp * self.iota0 - self.helicity_Np) / 
                        (self.helicity_N - self.helicity_M * self.iota0)
                    )
                )
            )
        return Jchi0_analytic

    def getJChiAnalytic1stOrder(self, peta):
        r"""
        Computes the 1st order analytic action derived in the NAE, \rho_\star
        expansion of the canonical momenta.

        Args:
            peta (float): 0th order Peta value used in calculation

        Returns:
            Jchi1_analytic (float): 1st order J_\chi
        """
        psi0 = (self.jac / (self.helicity_N - self.helicity_M*self.iota0)) * peta
        B_norm = self.m_norm / (self.q * self.T_norm)
        Bbar = self.B0 / B_norm
        G_norm = (self.m_norm * self.L_norm) / (self.q* self.T_norm)
        Gbar = self.G0 / G_norm
        A1 = np.sqrt(2)*Gbar
        A2 = Bbar * (self.helicity_M*self.iota0 - self.helicity_N)
        A3 = np.sqrt(2 * psi0 / Bbar)

        # check_expression_conditionals(A1, A2, A3)

        Jchi1_analytic = A1*(2*np.pi)/(A2*np.sqrt(1 - A3**2))
        return Jchi1_analytic / (2*np.pi)

    def getJChiAnalyticFiniteMu(self, peta):
        psi0 = (self.jac / (self.helicity_N - self.helicity_M*self.iota0)) * peta
        B_norm = self.m_norm / (self.q * self.T_norm)
        Bbar = self.B0 / B_norm
        mu_norm = (self.q * self.L_norm**2) / self.T_norm
        mubar_max = 1 / Bbar

        print(f"Bbar = {Bbar}")
        print(f"mubar_max = {mubar_max}")

        A2 = 0.9*mubar_max * Bbar
        A3 = np.sqrt(2 * psi0 / Bbar)

        check_expression_conditionals_finite_mu(A2, A3)

    def initNormalizedMomentaMaps(self, nLevels):
        """
        Computes 2D arrays of the momenta (both term by term and the full 
        expression). 

        Args: 
            nLevels (int): The number of levels to compute contours of momenta,
                which can be used to ask for momenta at specified contour
                levels. This is useful to overplot the root solve points on top
                of a contour in the plot_figures function.
        """
        self.vpar_init = np.full(self.ns*self.nchi, self.vpar0)

        pchi_term1, pchi_term2, peta_term1, peta_term2 = self.compute_normalized_momenta(
            points = self.points, 
            vpar = self.vpar_init,
            terms = True
        )

        self.pchi_term1 = pchi_term1.reshape(self.ns, self.nchi)
        self.pchi_term2 = pchi_term2.reshape(self.ns, self.nchi)
        self.peta_term1 = peta_term1.reshape(self.ns, self.nchi)
        self.peta_term2 = peta_term2.reshape(self.ns, self.nchi)

        self.peta_mat = self.peta_term1 + self.peta_term2
        self.pchi_mat = self.pchi_term1 + self.pchi_term2

        self.p_eta_spline_dict = {}
        fig, ax = plt.subplots(1, 1)
        im = ax.contour(
            self.chi_mat, self.s_mat, self.peta_mat, levels=nLevels
        )
        for level, segments in zip(im.levels, im.allsegs):
            combined_points = np.vstack(segments)

            if combined_points.size == 0:
                continue

            chi_coords = combined_points[:, 0]
            s_coords = combined_points[:, 1]

            sorted_chi, unique_indices = np.unique(chi_coords, return_index=True)
            sorted_s = s_coords[unique_indices]
            
            # Create and store the interpolant
            self.p_eta_spline_dict[level] = CubicSpline(sorted_chi, sorted_s)

        plt.close()

        self.keys = np.array(list(self.p_eta_spline_dict.keys()))

    def plot_figures(self, nLevels):
        """
        Creates and saves plots of 
            1. The individual terms of P_\chi and their ratio
            2. The magnetic field strength |B|
            3. The momentum contours for P_\chi and P_\eta, as well as
                overplots the points used in the numerical action integration if
                already calculated.

        Args:
            nLevels (int): Number of levels for the contour plot
        """
        # Plot momenta map terms
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        im0 = ax[0].pcolormesh(self.chi_mat, self.s_mat, self.pchi_term1)
        plt.colorbar(im0, ax=ax[0], label=r"$\bar{\rho_\parallel} M' \bar{G} $")
        im1 = ax[1].pcolormesh(self.chi_mat, self.s_mat, self.pchi_term2)
        plt.colorbar(im1, ax=ax[1], label=r"$\bar{\psi} (M' \iota_0 - N')$")
        im2 = ax[2].pcolormesh(self.chi_mat, self.s_mat, np.abs(self.pchi_term1 / self.pchi_term2), vmax = 0.3)
        plt.colorbar(im2, ax=ax[2], label=r"$(\bar{\rho_\parallel} M' \bar{G} ) / (\bar{\psi} (M' \iota_0 - N'))$")
        ax[0].set_xlabel("$\chi$")
        ax[1].set_xlabel("$\chi$")
        ax[2].set_xlabel("$\chi$")
        ax[0].set_ylabel("$s$")
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, self.subdir, "pchi_terms_normalized.pdf"))
        plt.close()

        # Plot modB
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        self.field.set_points(self.points)
        self.modB = self.field.modB()[:, 0].reshape(self.ns, self.nchi)
        im0 = ax[0].pcolormesh(self.chi_mat, self.s_mat, self.modB)
        plt.colorbar(im0, ax=ax[0], label="$|B|$")
        im1 = ax[1].contourf(self.chi_mat, self.s_mat, self.modB)
        plt.colorbar(im1, ax=ax[1], label="$|B|$")
        ax[0].set_xlabel("$\chi$")
        ax[1].set_xlabel("$\chi$")
        ax[0].set_ylabel("s")
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, self.subdir, "modB.pdf"))
        plt.close()

        # Plot momenta map s vs. chi with contours
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        chi_cs = ax[0].contour(self.chi_mat, self.s_mat, self.pchi_mat, levels=nLevels, cmap='rainbow')
        plt.colorbar(chi_cs, ax=ax[0], label="$P_\chi$")
        eta_cs = ax[1].contour(self.chi_mat, self.s_mat, self.peta_mat, levels=nLevels, cmap='rainbow')
        plt.colorbar(eta_cs, ax=ax[1], label="$P_\eta$")

        if hasattr(self, 'ss_plot'):
            ax[1].scatter(self.chis_plot, self.ss_plot, marker='*', color='k')

        for path in chi_cs.get_paths():
            if len(path.vertices) > 0:
                s_start = path.vertices[:, 1].min()
                ax[0].axhline(y=s_start, color='k', linestyle='--', alpha=0.5)

        for path in eta_cs.get_paths():
            if len(path.vertices) > 0:
                s_start = path.vertices[:, 1].min()
                ax[1].axhline(y=s_start, color='k', linestyle='--', alpha=0.5)

        ax[0].set_xlabel("$\chi$")
        ax[1].set_xlabel("$\chi$")
        ax[0].set_ylabel("s")
        fig.tight_layout()
        fig.savefig(os.path.join(self.outdir, self.subdir, "momenta_mapping_contour.pdf"))
        plt.close()