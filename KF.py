import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# --- 1. The Kalman Filter Class ---
class KalmanFilter2D:
    def __init__(self, dt, q_scale, r_scale):
        self.dt = dt
        # State vector: [x, y, velocity_x, velocity_y]
        self.x = np.zeros((4, 1))
        # Covariance Matrix
        self.P = np.eye(4)
        
        # State Transition Matrix (Physics model: x = x + v*dt)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Measurement Matrix (We only measure x and y, not velocity)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        self.set_Q(q_scale)
        self.set_R(r_scale)

    def set_Q(self, scale):
        # Process Noise Covariance
        self.Q = np.eye(4) * scale

    def set_R(self, scale):
        # Measurement Noise Covariance
        self.R = np.eye(2) * scale

    def predict(self):
        # Predict the next state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        # Update the state based on the noisy measurement
        z = np.array(z).reshape((2, 1))
        y = z - self.H @ self.x # Residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].flatten()

# --- 2. Setup the Simulation ---
dt = 0.05
t = np.arange(0, 100, dt)

# Generate a "True" path (a figure-8 / Lissajous curve)
true_x = np.sin(t) * 10
true_y = np.sin(2 * t) * 10

# Initialize the filter
kf = KalmanFilter2D(dt=dt, q_scale=0.01, r_scale=5.0)
kf.x[0, 0] = true_x[0] # Set initial x
kf.x[1, 0] = true_y[0] # Set initial y

# --- 3. Setup Matplotlib Figure and Sliders ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.25) # Make room for sliders
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_title("2D Kalman Filter - TensorRT Joint Tracker Simulation")
ax.grid(True)

# Plot elements
true_path_line, = ax.plot([], [], 'k--', alpha=0.3, label='True Path')
noisy_scatter = ax.scatter([], [], c='red', alpha=0.5, marker='x', label='Noisy TensorRT Measurement')
estimated_line, = ax.plot([], [], 'b-', linewidth=2, label='Kalman Filter Estimate')
estimated_point, = ax.plot([], [], 'bo', markersize=8)

ax.legend(loc='upper right')

# Slider axes
axcolor = 'lightgoldenrodyellow'
ax_R = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_Q = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)

# Create sliders
s_R = Slider(ax_R, 'Measurement Noise (R)', 0.1, 20.0, valinit=5.0)
s_Q = Slider(ax_Q, 'Process Noise (Q)', 0.001, 1.0, valinit=0.01)

# Update filter parameters when sliders change
def update_params(val):
    kf.set_R(s_R.val)
    kf.set_Q(s_Q.val)
s_R.on_changed(update_params)
s_Q.on_changed(update_params)

# --- 4. The Animation Loop ---
frame_idx = 0
history_x, history_y = [], []
meas_hist_x, meas_hist_y = [], []

def animate(i):
    global frame_idx
    
    # Reset history if we loop the animation
    if frame_idx >= len(t):
        frame_idx = 0
        history_x.clear()
        history_y.clear()
        meas_hist_x.clear()
        meas_hist_y.clear()
        kf.x[0,0] = true_x[0]
        kf.x[1,0] = true_y[0]

    # Get True Position
    tx = true_x[frame_idx]
    ty = true_y[frame_idx]
    
    # Generate Noisy Measurement (Current R value dictates scatter size)
    noise_std = np.sqrt(s_R.val)
    mx = tx + np.random.normal(0, noise_std)
    my = ty + np.random.normal(0, noise_std)
    
    # Apply Kalman Filter
    kf.predict()
    ex, ey = kf.update([mx, my])
    
    # Keep track of recent history for plotting trails
    history_x.append(ex)
    history_y.append(ey)
    meas_hist_x.append(mx)
    meas_hist_y.append(my)
    
    # Keep trails relatively short
    max_trail = 50
    if len(history_x) > max_trail:
        history_x.pop(0)
        history_y.pop(0)
        meas_hist_x.pop(0)
        meas_hist_y.pop(0)

    # Update visual elements
    true_path_line.set_data(true_x, true_y)
    noisy_scatter.set_offsets(np.c_[meas_hist_x, meas_hist_y])
    estimated_line.set_data(history_x, history_y)
    estimated_point.set_data([ex], [ey])
    
    frame_idx += 1
    return true_path_line, noisy_scatter, estimated_line, estimated_point

ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()