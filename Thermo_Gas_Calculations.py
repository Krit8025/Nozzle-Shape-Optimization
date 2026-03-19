import numpy as np
import math
import matplotlib.pyplot as plt

class GasDynamics:
    def __init__(self, gamma=1.4):
        self.gamma = gamma
        self.gp1 = self.gamma + 1.0
        self.gm1 = self.gamma - 1.0
        self.g_ratio = math.sqrt(self.gp1 / self.gm1)
#mach angle caluclator
    def get_mu(self, M):
        if M <= 1.0:
            return math.pi / 2.0  # 90 degrees at sonic conditions
        return math.asin(1.0 / M)
# nu calculator
    def get_nu(self, M):
        """Returns the Prandtl-Meyer angle (in radians) for a given Mach number."""
        if M <= 1.0:
            return 0.0

        M2_minus_1 = M ** 2 - 1.0
        term1 = self.g_ratio * math.atan(math.sqrt((self.gm1 / self.gp1) * M2_minus_1))
        term2 = math.atan(math.sqrt(M2_minus_1))

        return term1 - term2
# derivatie I use for Newton rasphon
    def get_dnu_dM(self, M):
        if M <= 1.0:
            return 0.0  # Mathematically zero at M=1

        numerator = math.sqrt(M ** 2 - 1.0)
        denominator = M * (1.0 + (self.gm1 / 2.0) * M ** 2)

        return numerator / denominator
# newton rasphon for getting Mach number from nu
    def get_M_from_nu(self, target_nu, tolerance=1e-6, max_iterations=50):
        #Uses Newton-Raphson to find Mach number for a given Prandtl-Meyer angle.
        # SAFETY NET: Max physical nu for gamma=1.4 is ~2.27 rad (130 degrees)
        # Cap it slightly below the limit to prevent M -> infinity
        max_nu = 2.25
        if target_nu >= max_nu:
            return 20.0  # Return a very high Mach number safely, rather than exploding
        if target_nu <= 0.0:
            return 1.0  # Cannot drop below Mach 1
        if target_nu <= 0.0:
            return 1.0

        #initially guess
        M = 2.0
        for _ in range(max_iterations):
            current_nu = self.get_nu(M)
            error = current_nu - target_nu
            if abs(error) < tolerance:
                return M
            derivative = self.get_dnu_dM(M)
            if derivative == 0:
                break

            M = M - (error / derivative)
            M = max(1.0001, M)

        return M
"""
# test for gas dynamics
aero = GasDynamics(gamma=1.4)
test_M = 2.4
calc_nu_rad = aero.get_nu(test_M)
calc_mu_rad = aero.get_mu(test_M)
calc_nu_deg = math.degrees(calc_nu_rad)
calc_mu_deg = math.degrees(calc_mu_rad)
print(f"getting nu and mu from M")
print(f"Input Mach     : {test_M}")
print(f"Calculated nu  : {calc_nu_deg:.3f} degrees")
print(f"Calculated mu  : {calc_mu_deg:.3f} degrees\n")
recovered_M = aero.get_M_from_nu(calc_nu_rad)

print(f"inverse, using newton Rasphon")
print(f"Input nu       : {calc_nu_deg:.3f} degrees")
print(f"Recovered Mach : {recovered_M:.6f}")

if abs(test_M - recovered_M) < 1e-5:
    print("SUCCESS: Inverse function perfectly recovers the Mach number!")
else:
    print("WARNING: Newton-Raphson failed to converge accurately.")
"""