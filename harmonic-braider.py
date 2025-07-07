from __future__ import annotations
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys

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

"""Update parameter value and redraw when a slider is moved."""
class BraiderApp(QtWidgets.QWidget):
    """
    A qt-based application embedding a Matplotlib 3D plot.

    Controls:
      - H1, H2, H3: fundamental frequencies of the harmonics
      - B: braiding amplitude
      - S: sub-harmonic amplitude
      - Twist Factor: Möbius twisting rate
      - Quat X/Y/Z: quaternion vector part for rotation
      - Zoom: scale of the 3D axes
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Harmonic Braiding Visualizer (PyQt6)")
        self.resize(1000, 600)

        # Parameter definitions with labels, ranges, and initial values
        self.param_specs = {
            'H1': {'label': 'H1 (Harmonic 1)', 'min': 5.0, 'max': 15.0, 'step': 0.01, 'init': 9.71},
            'H2': {'label': 'H2 (Harmonic 2)', 'min': 5.0, 'max': 15.0, 'step': 0.01, 'init': 7.55},
            'H3': {'label': 'H3 (Harmonic 3)', 'min': 5.0, 'max': 15.0, 'step': 0.01, 'init': 8.63},
            'B':  {'label': 'B (Braiding Amp)', 'min': 0.001, 'max': 2.0, 'step': 0.001, 'init': 0.01},
            'S':  {'label': 'S (Sub-harmonic Amp)', 'min': -5.0, 'max': 5.0, 'step': 0.001, 'init': 0.01},
            'twist': {'label': 'Twist Factor', 'min': -2.0, 'max': 2.0, 'step': 0.01, 'init': 0.1},
            'qx': {'label': 'Quat X', 'min': -1.0, 'max': 1.0, 'step': 0.001, 'init': 0.0},
            'qy': {'label': 'Quat Y', 'min': -1.0, 'max': 1.0, 'step': 0.001, 'init': 0.0},
            'qz': {'label': 'Quat Z', 'min': -1.0, 'max': 1.0, 'step': 0.001, 'init': 0.0},
            'zoom': {'label': 'Zoom', 'min': 0.1, 'max': 10.0, 'step': 0.1, 'init': 1.0},
        }

        # Extract initial parameter values
        self.params = {name: spec['init'] for name, spec in self.param_specs.items()}
        # Time samples for the curve
        self.t = np.linspace(0, 10, 800)

        layout = QtWidgets.QHBoxLayout(self)
        control_panel = QtWidgets.QVBoxLayout()

        self.sliders = {}
        for key, spec in self.param_specs.items():
            label = QtWidgets.QLabel(spec['label'])
            slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(int((spec['max'] - spec['min']) / spec['step']))
            slider.setValue(int((spec['init'] - spec['min']) / spec['step']))
            slider.valueChanged.connect(lambda val, k=key: self.update_param(k, val))
            control_panel.addWidget(label)
            control_panel.addWidget(slider)
            self.sliders[key] = (slider, spec)

        layout.addLayout(control_panel)

        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.plot()

    def update_param(self, key, slider_val):
        """Update parameter value and redraw when a slider is moved."""
        spec = self.param_specs[key]
        self.params[key] = spec['min'] + slider_val * spec['step']
        self.plot()

    def plot(self) -> None:
        """Compute the braided curve and update the 3D plot."""
        # Build argument dict for geometry function, excluding Zoom
        geom_kwargs = {
            'H1': self.params['H1'],
            'H2': self.params['H2'],
            'H3': self.params['H3'],
            'B': self.params['B'],
            'S': self.params['S'],
            'twist': self.params['twist'],
            'qx': self.params['qx'],
            'qy': self.params['qy'],
            'qz': self.params['qz'],
        }

        # Generate curve
        xr, yr, zr = braided_curve(self.t, **geom_kwargs)

        # Clear and redraw
        self.ax.clear()
        self.ax.plot(xr, yr, zr, lw=2)

        # Apply zoom by scaling axis limits around the mean
        zf = self.params['zoom']
        rng = max(np.ptp(xr), np.ptp(yr), np.ptp(zr)) / (2 * zf)
        cx, cy, cz = np.mean(xr), np.mean(yr), np.mean(zr)
        self.ax.set_xlim(cx - rng, cx + rng)
        self.ax.set_ylim(cy - rng, cy + rng)
        self.ax.set_zlim(cz - rng, cz + rng)

        # Render in the GUI
        self.canvas.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = BraiderApp()
    window.show()
    sys.exit(app.exec())
