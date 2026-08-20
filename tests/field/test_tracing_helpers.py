import unittest
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import BoozerRadialInterpolant
from firm3d.field.tracing_helpers import (
    initialize_position_uniform_s_chi_grid
)

# TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
# TODO: Add proper file testing path
filename = "/Users/aveksler/work/repositories/firm3d/examples/inputs/boozmn_aten_rescaled.nc"

class TestTracingHelpers(unittest.TestCase):
    """Test tracing_helpers.py functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple Boozer field for testing
        self.field = BoozerRadialInterpolant(filename, 1, comm=None)
    
    def test_initialize_position_uniform_s_chi_grid(self):
        """
        Test initialize_position_uniform_s_chi_grid.
        Use case: User calls initialize_position_uniform_s_chi_grid with some
            number of points for s and chi.
        Expected: A list of (s, theta, zeta) points that correspond
            to a uniform distribution of particles in s and chi is returnde.
        """
        # Resolution in the s and chi directions
        ns = 5
        nchi = 5
        
        s = np.linspace(0, 1, ns)
        chi = np.linspace(0, 2*np.pi, nchi, endpoint=False)
        # The expected s array should repeat every entry nchi times
        s_expected = np.repeat(s, nchi)
        # The expected chi array is sorted for ease of comparison.
        chi_expected = np.sort(np.repeat(chi, ns))

        # Check for QA equilibria
        helicity_M = 1
        helicity_N = 0 # Doesn't matter as nfp isn't used for theta_grid
        points_output = initialize_position_uniform_s_chi_grid(
            self.field, ns, nchi, helicity_M=helicity_M, helicity_N=helicity_N
        )
        s_output = points_output[:, 0]
        theta_output = points_output[:, 1]
        zeta_output = points_output[:, 2]
        chi_output = helicity_M*theta_output - helicity_N*zeta_output
        chi_output = np.sort(np.mod(chi_output, 2*np.pi))
        assert np.allclose(s_expected, s_output)
        assert np.allclose(chi_expected, chi_output)

        # Check for QH equilibria
        helicity_M = 1
        helicity_N = self.field.nfp # Doesn't matter here as nfp isn't used for theta_grid
        points_output = initialize_position_uniform_s_chi_grid(
            self.field, ns, nchi, helicity_M=helicity_M, helicity_N=helicity_N
        )
        s_output = points_output[:, 0]
        theta_output = points_output[:, 1]
        zeta_output = points_output[:, 2]
        chi_output = helicity_M*theta_output - helicity_N*zeta_output
        chi_output = np.sort(np.mod(chi_output, 2*np.pi))
        assert np.allclose(s_expected, s_output)
        assert np.allclose(chi_expected, chi_output)

        # Check for QP equilibria
        helicity_M = 0
        helicity_N = self.field.nfp # helicity_N = 1 # Helicity matters here as nfp is used to define zeta grid.
        points_output = initialize_position_uniform_s_chi_grid(
            self.field, ns, nchi, helicity_M=helicity_M, helicity_N=helicity_N
        )
        s_output = points_output[:, 0]
        theta_output = points_output[:, 1]
        zeta_output = points_output[:, 2]
        chi_output = helicity_M*theta_output - helicity_N*zeta_output
        chi_output = np.sort(np.mod(chi_output, 2*np.pi))
        assert np.allclose(s_expected, s_output)
        assert np.allclose(chi_expected, chi_output)

if __name__ == "__main__":
    unittest.main()