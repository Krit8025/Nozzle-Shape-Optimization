import math


class Node:

    def __init__(self, x, y, M, theta, nu, mu):
        self.x = x
        self.y = y
        self.M = M
        self.theta = theta
        self.nu = nu
        self.mu = mu


def calculate_interior_point(node1, node2, aero: GasDynamics):
    """
    node1: Upstream node on the C- characteristic (Right-running)
    node2: Upstream node on the C+ characteristic (Left-running)
    """

    inv_minus = node1.theta + node1.nu  # Invariant from node 1
    inv_plus = node2.theta - node2.nu  # Invariant from node 2

    theta3 = 0.5 * (inv_minus + inv_plus)
    nu3 = 0.5 * (inv_minus - inv_plus)

    # Use our GasDynamics engine to get the rest of the properties!
    M3 = aero.get_M_from_nu(nu3)
    mu3 = aero.get_mu(M3)

    # 2. Geometry: Solve for x3 and y3 using average slopes
    # Angle of the C- line (node 1 to 3)
    angle_minus = 0.5 * ((node1.theta - node1.mu) + (theta3 - mu3))
    m_minus = math.tan(angle_minus)

    # Angle of the C+ line (node 2 to 3)
    angle_plus = 0.5 * ((node2.theta + node2.mu) + (theta3 + mu3))
    m_plus = math.tan(angle_plus)

    # Prevent divide-by-zero if lines are somehow parallel (rare but safe)
    if abs(m_plus - m_minus) < 1e-6:
        raise ValueError("Characteristics are parallel, cannot find intersection!")

    # Intersection formula
    x3 = (node1.y - node2.y - m_minus * node1.x + m_plus * node2.x) / (m_plus - m_minus)
    y3 = node1.y + m_minus * (x3 - node1.x)

    # Return the newly calculated Node
    return Node(x3, y3, M3, theta3, nu3, mu3)