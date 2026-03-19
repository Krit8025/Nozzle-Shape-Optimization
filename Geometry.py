import numpy as np
import math
import matplotlib.pyplot as plt

class NozzleWall:
    def __init__(self, P0, P1, P2, P3):
        self.x0, self.y0 = P0
        self.x1, self.y1 = P1
        self.x2, self.y2 = P2
        self.x3, self.y3 = P3
        #define the bezier curve for 3 points
    def get_point(self, t):
        t = max(0.0, min(t, 1.0))
        x = ((1 - t) ** 3 * self.x0 +
             3 * (1 - t) ** 2 * t * self.x1 +
             3 * (1 - t) * t ** 2 * self.x2 +
             t ** 3 * self.x3)
        y = ((1 - t) ** 3 * self.y0 +
             3 * (1 - t) ** 2 * t * self.y1 +
             3 * (1 - t) * t ** 2 * self.y2 +
             t ** 3 * self.y3)
        return x, y

    def get_derivatives(self, t):
        t = max(0.0, min(1.0, t))

        dxdt = (3 * (1 - t) ** 2 * (self.x1 - self.x0) +
                6 * (1 - t) * t * (self.x2 - self.x1) +
                3 * t ** 2 * (self.x3 - self.x2))

        dydt = (3 * (1 - t) ** 2 * (self.y1 - self.y0) +
                6 * (1 - t) * t * (self.y2 - self.y1) +
                3 * t ** 2 * (self.y3 - self.y2))

        return dxdt, dydt

    def get_theta(self, t):
        dxdt, dydt = self.get_derivatives(t)
        return math.atan2(dydt, dxdt)

    def get_t_from_x(self, x_target, tolerance=1e-6, max_iterations=50):
        if x_target <= self.x0: return 0.0
        if x_target >= self.x3: return 1.0

#assumING the curve is roughly linear in x
        t = (x_target - self.x0) / (self.x3 - self.x0)

        for _ in range(max_iterations):
            x_current, _ = self.get_point(t)
            error = x_current - x_target
            if abs(error) < tolerance:
                return t
            dxdt, _ = self.get_derivatives(t)
            if dxdt == 0:
                break

#Newton-Raphson step
            t = t - (error / dxdt)
            t = max(0.0, min(1.0, t))
        return t
    def draw(self, P0, P1, P2, P3):
        wall = NozzleWall(P0, P1, P2, P3)
        t_values = np.linspace(0, 1, 100)
        x_curve = []
        y_curve = []

        for t in t_values:
            x, y = wall.get_point(t)
            x_curve.append(x)
            y_curve.append(y)

        ctrl_x = [P0[0], P1[0], P2[0], P3[0]]
        ctrl_y = [P0[1], P1[1], P2[1], P3[1]]
        # test all the functions for an example of x point
        test_x = 2.0
        test_t = wall.get_t_from_x(test_x)
        actual_x, test_y = wall.get_point(test_t)
        test_theta_deg = math.degrees(wall.get_theta(test_t))

        print(f"Testing the inverse function at x = {test_x}:")
        print(f"Found t = {test_t:.4f}")
        print(f"Calculated coordinates: ({actual_x:.4f}, {test_y:.4f})")
        print(f"Wall angle (theta)  : {test_theta_deg:.2f} degrees")

        # plotting
        plt.figure(figsize=(10, 5))

        # Plot the centerline of the nozzle (axis of symmetry)
        plt.axhline(0, color='black', linestyle='-.', label='Centerline')

        # Plot the continuous wall curve
        plt.plot(x_curve, y_curve, 'b-', linewidth=2, label='Nozzle Wall Contour')

        # Plot the control points and the "control polygon" (dashed lines connecting them)
        plt.plot(ctrl_x, ctrl_y, 'ro--', alpha=0.5, label='Control Polygon')
        plt.scatter(ctrl_x, ctrl_y, color='red', s=50, zorder=5)

        # Highlight our test point
        plt.plot(test_x, test_y, 'go', markersize=8, label=f'Test Point (x={test_x})')

        # Formatting the plot
        plt.title('Parameterized CD Nozzle Divergent Section (Cubic Bézier)')
        plt.xlabel('Axial Distance (x)')
        plt.ylabel('Radial Distance (y)')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend()
        plt.axis('equal')  # This is CRUCIAL so angles look physically correct!
        plt.show()
        return 0
""""
P0 = (0.0, 1.0)
P1 = (1.5, 1.8)
P2 = (3.0, 2.5)
P3 = (5.0, 3.0) # Exit
"""

