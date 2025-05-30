# Harmonic Braiding • Tkinter GUI (Improved Sliders with Documentation)
#
# Mathematical Formulation:
#   x(t) = cos(H1*t) + B*cos(H2*t) + S*cos(H3*t)
#   y(t) = sin(H1*t) + B*sin(H2*t) + S*sin(H3*t)
#   z(t) = sin(B*t) + S*cos(B*t)
#
#   where:
#     H1 = number of base loops (cos/sin cycles over t∈[0,10])
#     H2 = number of braid wraps (secondary frequency)
#     H3 = number of sub-harmonic ripples
#     B  = amplitude for the H2 component
#     S  = amplitude for the H3 component
#
#   A Möbius twist is applied in-plane:
#     θ(t) = twist * t
#     [x,y] → rotation by θ(t)
#
#   Finally, a quaternion [qx, qy, qz, qw=1] is normalized and applied to (x,y,z)
#   to reorient the curve in 3D.
#
#   © 2025 iowyth hezel ulthiin

from __future__ import annotations
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for embedding Matplotlib in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# -------- geometry ----------------------------------------------------
def braided_curve(
    t: np.ndarray,
    H1: float,  # frequency of the first harmonic (base cosine/sine)
    H2: float,  # frequency of the second harmonic (braiding component)
    H3: float,  # frequency of the third harmonic (subtle modulation)
    B: float,   # amplitude of the braiding harmonic (multiplies cos/sin of H2)
    S: float,   # amplitude of the sub-harmonic (multiplies cos/sin of H3)
    twist: float,  # Möbius twist rate (applied as rotation in XY-plane)
    qx: float,  # X component of quaternion rotation
    qy: float,  # Y component of quaternion rotation
    qz: float,  # Z component of quaternion rotation
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 3D coordinates of the braided curve.

    - x, y: sum of three sinusoids with optional Möbius twist in the XY plane.
    - z: combination of a sine and cosine to lift the braid in Z.
    - Apply quaternion rotation to orient the braid in 3D space.

    Returns:
        (xr, yr, zr): rotated and twisted coordinates.
    """
    # Base curve: superposition of three rotating sinusoids
    x = np.cos(H1 * t) + B * np.cos(H2 * t) + S * np.cos(H3 * t)
    y = np.sin(H1 * t) + B * np.sin(H2 * t) + S * np.sin(H3 * t)
    z = np.sin(B * t) + S * np.cos(H3 * t)  # use H3 for sub-harmonic ripples in Z

    # Möbius twist: rotate XY by angle = twist * t
    theta = twist * t
    x, y = (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta),
    )

    # Build and normalize quaternion (qw fixed to 1)
    quat = np.array([qx, qy, qz, 1.0])
    quat /= np.linalg.norm(quat)

    # Apply quaternion to all points
    pts = np.vstack((x, y, z)).T  # shape (N, 3)
    xr, yr, zr = R.from_quat(quat).apply(pts).T  
    return xr, yr, zr

# -------- application -------------------------------------------------
class BraiderApp:
    """
    A Tkinter-based application embedding a Matplotlib 3D plot.

    Controls:
      - H1, H2, H3: fundamental frequencies of the harmonics
      - B: braiding amplitude
      - S: sub-harmonic amplitude
      - Twist Factor: Möbius twisting rate
      - Quat X/Y/Z: quaternion vector part for rotation
      - Zoom: scale of the 3D axes
    """

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title('Harmonic Braiding Visualizer')

        # Parameter definitions with labels, ranges, and initial values
        self.param_specs = {
            'H1 (Harmonic 1)': {'min': 5.0,  'max': 15.0, 'step': 0.01,  'init': 9.71},
            'H2 (Harmonic 2)': {'min': 5.0,  'max': 15.0, 'step': 0.01,  'init': 7.55},
            'H3 (Harmonic 3)': {'min': 5.0,  'max': 15.0, 'step': 0.01,  'init': 8.63},
            'B (Braiding Amp)': {'min': 0.001,'max':  2.0, 'step': 0.001, 'init': 0.01},
            'S (Sub-harmonic Amp)': {'min': -5.0, 'max':  5.0, 'step': 0.001, 'init': 0.01},
            'Twist Factor':      {'min': -2.0, 'max':  2.0, 'step': 0.01,  'init': 0.1},
            'Quat X':            {'min': -1.0, 'max':  1.0, 'step': 0.001, 'init': 0.0},
            'Quat Y':            {'min': -1.0, 'max':  1.0, 'step': 0.001, 'init': 0.0},
            'Quat Z':            {'min': -1.0, 'max':  1.0, 'step': 0.001, 'init': 0.0},
            'Zoom':              {'min':  0.1, 'max': 10.0, 'step': 0.1,  'init': 1.0},
        }

        # Extract initial parameter values
        self.params = {name: spec['init'] for name, spec in self.param_specs.items()}
        # Time samples for the curve
        self.t = np.linspace(0, 10, 800)

        # Create Matplotlib Figure and 3D axes in the GUI
        self.fig = plt.Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Build control panel on the right
        ctrl_frame = ttk.Frame(master)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # Create one slider per parameter
        self.sliders: dict[str, tk.Scale] = {}
        for name, spec in self.param_specs.items():
            ttk.Label(ctrl_frame, text=name).pack(anchor='w')
            slider = tk.Scale(
                ctrl_frame,
                from_=spec['min'], to=spec['max'],
                resolution=spec['step'], orient=tk.HORIZONTAL,
                length=200,
                command=lambda val, n=name: self.on_slider(n, val)
            )
            slider.set(spec['init'])
            slider.pack(fill=tk.X, pady=2)
            self.sliders[name] = slider

        # Initial draw
        self.plot()

    def on_slider(self, name: str, val: str) -> None:
        """Update parameter value and redraw when a slider is moved."""
        self.params[name] = float(val)
        self.plot()

    def plot(self) -> None:
        """Compute the braided curve and update the 3D plot."""
        # Build argument dict for geometry function, excluding Zoom
        geom_kwargs = {
            'H1': self.params['H1 (Harmonic 1)'],
            'H2': self.params['H2 (Harmonic 2)'],
            'H3': self.params['H3 (Harmonic 3)'],
            'B':  self.params['B (Braiding Amp)'],
            'S':  self.params['S (Sub-harmonic Amp)'],
            'twist': self.params['Twist Factor'],
            'qx': self.params['Quat X'],
            'qy': self.params['Quat Y'],
            'qz': self.params['Quat Z'],
        }

        # Generate curve
        xr, yr, zr = braided_curve(self.t, **geom_kwargs)

        # Clear and redraw
        self.ax.clear()
        self.ax.plot(xr, yr, zr, lw=2)

        # Apply zoom by scaling axis limits around the mean
        zf = self.params['Zoom']
        rng = max(np.ptp(xr), np.ptp(yr), np.ptp(zr)) / (2 * zf)
        cx, cy, cz = np.mean(xr), np.mean(yr), np.mean(zr)
        self.ax.set_xlim(cx - rng, cx + rng)
        self.ax.set_ylim(cy - rng, cy + rng)
        self.ax.set_zlim(cz - rng, cz + rng)

        # Render in the GUI
        self.canvas.draw()

if __name__ == '__main__':
    # Launch the application
    root = tk.Tk()
    app = BraiderApp(root)
    root.mainloop()
