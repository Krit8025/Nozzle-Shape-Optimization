import math
import numpy as np
from scipy.optimize import brentq
from Geometry import NozzleWall
from Thermo_Gas_Calculations import GasDynamics

class Node:

    def __init__(self, x, y, M, theta, nu, mu):
        self.x = x
        self.y = y
        self.M = M
        self.theta = theta
        self.nu = nu
        self.mu = mu


def calculate_interior_point(node_up, node_down, aero):
    # Slopes of characteristics
    slope_plus = np.tan(node_down.theta + node_down.mu)
    slope_minus = np.tan(node_up.theta - node_up.mu)

    # 1. Calculate physical location (x, y)
    x3 = (node_up.y - node_down.y + slope_plus * node_down.x - slope_minus * node_up.x) / (slope_plus - slope_minus)
    y3 = node_down.y + slope_plus * (x3 - node_down.x)

    # 2. Correct 2D Compatibility Equations
    # C- comes from node_up (travels down) -> Invariant: theta + nu
    I_minus = node_up.theta + node_up.nu
    # C+ comes from node_down (travels up) -> Invariant: theta - nu
    I_plus = node_down.theta - node_down.nu

    # 3. Solve for new properties
    theta3 = 0.5 * (I_minus + I_plus)
    nu3 = 0.5 * (I_minus - I_plus)

    return Node(x=x3, y=y3, M=aero.get_M_from_nu(nu3), theta=theta3, nu=nu3,
                mu=aero.get_mu(aero.get_M_from_nu(nu3)))


def calculate_centerline_point(node_up, aero):
    # C- wave travels down to hit the centerline
    slope_minus = np.tan(node_up.theta - node_up.mu)

    x3 = node_up.x - (node_up.y / slope_minus)
    y3 = 0.0

    # Centerline forces flow angle to be zero
    theta3 = 0.0

    # C- Invariant: theta + nu = constant
    I_minus = node_up.theta + node_up.nu
    nu3 = I_minus - theta3  # Since theta3 is 0, nu3 just equals I_minus

    return Node(x=x3, y=y3, M=aero.get_M_from_nu(nu3), theta=theta3, nu=nu3,
                mu=aero.get_mu(aero.get_M_from_nu(nu3)))


def calculate_wall_point(node_down, wall, aero):
    """
    Calculates the 2D planar wall point by intersecting the C+ characteristic
    from node_down with the parametric Bezier wall.
    """
    # 1. Slope of the C+ wave traveling up towards the wall
    slope_plus = np.tan(node_down.theta + node_down.mu)

    # 2. Define the intersection error function based on Bezier parameter t
    def intersection_error(t):
        # Get the (x, y) on the wall for a given t
        x_wall, y_wall = wall.get_point(t)

        # Line equation from node_down: y - y_down = slope * (x - x_down)
        # We calculate where the straight C+ line would be at x_wall
        y_line = node_down.y + slope_plus * (x_wall - node_down.x)

        # The error is the vertical distance between the wall and the C+ line
        return y_wall - y_line

    # 3. Find the parameter t where the error is zero (intersection point)
    # The intersection must be downstream of node_down. We'll search from 0 to 1.
    try:
        # brentq is a highly robust root-finding algorithm
        t_int = brentq(intersection_error, 0.0, 1.0)
    except ValueError:
        # Fallback linear search if the exact bracket isn't immediately obvious
        t_vals = np.linspace(0.0, 1.0, 100)
        errors = [intersection_error(t) for t in t_vals]
        t_int = 1.0  # Default to exit if it fails
        for i in range(len(errors) - 1):
            if errors[i] * errors[i + 1] <= 0:
                t_int = brentq(intersection_error, t_vals[i], t_vals[i + 1])
                break

    # 4. Extract exact intersection coordinates
    x3, y3 = wall.get_point(t_int)

    # 5. Calculate the local wall angle (theta3) using a finite difference step
    dt = 0.0001
    if t_int + dt <= 1.0:
        x_next, y_next = wall.get_point(t_int + dt)
        theta3 = np.arctan2(y_next - y3, x_next - x3)
    else:
        x_prev, y_prev = wall.get_point(t_int - dt)
        theta3 = np.arctan2(y3 - y_prev, x3 - x_prev)

    # 6. Correct 2D C+ Invariant: theta - nu = constant
    I_plus = node_down.theta - node_down.nu

    # Gas cannot flow through metal, so flow angle must perfectly match wall angle
    # Solve for nu3 using the invariant
    nu3 = theta3 - I_plus

    # 7. Get matching thermodynamic properties
    M3 = aero.get_M_from_nu(nu3)
    mu3 = aero.get_mu(M3)

    return Node(x=x3, y=y3, M=M3, theta=theta3, nu=nu3, mu=mu3)

"""
# --- TESTING THE WALL POINT CALCULATION ---

# 1. Initialize our engines
aero = GasDynamics(gamma=1.4)

# Re-create our divergent nozzle wall from earlier
P0 = (0.0, 1.0)
P1 = (0.5, 1.8)
P2 = (3.0, 2.5)
P3 = (5.0, 2.5)
wall = NozzleWall(P0, P1, P2, P3)

# 2. Create Node 2 (Our upstream point inside the flow)
# Let's place it at x=1.0, y=0.5 (well below the wall)
# Assume the flow here is Mach 2.0, angled slightly upwards at 5 degrees
test_M2 = 2.0
test_theta2 = math.radians(5.0)
test_nu2 = aero.get_nu(test_M2)
test_mu2 = aero.get_mu(test_M2)

node2 = Node(x=1.0, y=0.5, M=test_M2, theta=test_theta2, nu=test_nu2, mu=test_mu2)

# 3. Calculate the Wall Point (Node 3)
print("Calculating Wall Point... (Running Predictor-Corrector Loop)")
node3 = calculate_wall_point(node2, wall, aero)

# 4. Verification checks
# Let's ask the wall exactly what its properties are at the new x3 location
t_check = wall.get_t_from_x(node3.x)
_, expected_y = wall.get_point(t_check)
expected_theta = wall.get_theta(t_check)

print("\n--- WALL POINT RESULTS ---")
print(f"Intersection Location : x = {node3.x:.4f}, y = {node3.y:.4f}")
print(f"Calculated Mach       : {node3.M:.4f}")
print(f"Calculated Flow Angle : {math.degrees(node3.theta):.4f} degrees")
print(f"Calculated Nu         : {math.degrees(node3.nu):.4f} degrees")

print("\n--- SANITY CHECKS ---")
# Check 1: Did it actually hit the wall?
y_error = abs(node3.y - expected_y)
print(f"Geometry Check: y-coordinate error is {y_error:.8f}")
if y_error < 1e-5:
    print("  -> SUCCESS: Point lies perfectly on the Bezier curve.")

# Check 2: Does the flow follow the wall?
theta_error = abs(node3.theta - expected_theta)
print(f"Boundary Check: Flow angle error is {theta_error:.8f} radians")
if theta_error < 1e-5:
    print("  -> SUCCESS: Flow is perfectly parallel to the physical wall.")
    """