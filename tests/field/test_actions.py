import unittest
from pathlib import Path

import numpy as np

from firm3d.field.actions import JchiAction
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from tempfile import NamedTemporaryFile

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()
filename_jchi_error = str(
    (TEST_DIR / "jchi_action_error_boozanalytic.dat").resolve()
)

class TestJchiAction(unittest.TestCase):
    """
    Test the JchiAction class at mu = 0 by comparing the numerical, 0th order
    analytical, and 1st order analytical actions to a test file 
    """
    def test_jchi(self):
        # Initialize reactor / field paratmers
        R = 10
        a = 1
        B0 = 10
        iota0 = 0.5

        # Set alpha particle parameters
        Ekin = FUSION_ALPHA_PARTICLE_ENERGY
        mass = ALPHA_PARTICLE_MASS
        q = ALPHA_PARTICLE_CHARGE

        jchi_n_arr = []
        jchi0_a_arr = []
        jchi1_a_arr = []

        Action = JchiAction(
            Ekin=Ekin,
            mass=mass,
            q=q,
        )
        # Initialize magnetic field
        Action.initField(
            B0=B0,
            R=R,
            a=a,
            iota0=iota0,
            helicity_M=1,
            helicity_N=4,
            helicity_Mp=1,
            helicity_Np=-2,
        )

        s = 0.4
        psi_norm = (Action.m_norm * Action.L_norm**2) / (Action.q * Action.T_norm)
        psi = (Action.field.psi0/psi_norm) * s
        peta = (1 / (Action.jac)) * (psi * (Action.helicity_N - Action.helicity_M * Action.iota0))

        # Compute and save Jchi
        Action.getJchi(peta=peta, nchi=int(1e3))

        jchi_n_arr.append(Action.JChi_numeric)
        jchi0_a_arr.append(Action.JChi0_analytic)
        jchi1_a_arr.append(Action.JChi1_analytic)

        data = np.column_stack(
            (
                np.array(jchi_n_arr),
                np.array(jchi0_a_arr),
                np.array(jchi1_a_arr)
            )
        )

        temp_file = self.enterContext(NamedTemporaryFile(mode="w+", suffix=".dat"))

        np.savetxt(temp_file.name, data, delimiter=",")
    
        # Rewind to the beginning to read it back
        temp_file.seek(0)
        content = temp_file.read()

        with open(filename_jchi_error, "r") as f:
            expected_text = f.read()
        
        # Assertions
        self.assertEqual(content, expected_text)


if __name__ == "__main__":
    unittest.main()