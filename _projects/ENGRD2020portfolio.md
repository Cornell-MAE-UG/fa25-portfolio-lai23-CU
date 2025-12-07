```md
---
layout: project
title: ENGRD 2020 Portfolio
image: /assets/img/engrd2020portfolio.png
permalink: /ENGRD2020portfolio/
---

```python
import numpy as np
import matplotlib.pyplot as plt

# Design space (cm)
Lx, Ly = 150.0, 50.0

# Actuator specs (IMA55 RN05, strongest option)
F_actuator_max = 8050  # N peak thrust

# Sweep joint heights
y_vals = np.linspace(5, Ly, 200)
W_vals = []

for y_joint in y_vals:
    x_joint = np.sqrt(max(0, (Lx/2)**2 - (y_joint/2)**2))
    beam_len = np.sqrt(x_joint**2 + y_joint**2)
    act_len = np.sqrt((Lx - x_joint)**2 + y_joint**2)

    if act_len == 0: 
        W_vals.append(0)
        continue

    theta = np.arcsin(y_joint / act_len)
    L_ratio = beam_len / y_joint
    W_max = F_actuator_max * (L_ratio / np.sin(theta))
    W_vals.append(W_max)

W_vals = np.array(W_vals)

# Normalize for tradeoff analysis
h_norm = (y_vals - y_vals.min()) / (y_vals.max() - y_vals.min())
w_norm = (W_vals - W_vals.min()) / (W_vals.max() - W_vals.min())

p1 = np.array([h_norm[0], w_norm[0]])
p2 = np.array([h_norm[-1], w_norm[-1]])

def point_line_dist(p, a, b):
    return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

distances = [point_line_dist(np.array([h, w]), p1, p2) 
             for h, w in zip(h_norm, w_norm)]
opt_index = np.argmax(distances)

opt_height = y_vals[opt_index]
opt_weight = W_vals[opt_index]

print("=== Optimized Height–Weight Tradeoff ===")
print(f"  Optimal lift height: {opt_height:.2f} cm")
print(f"  Optimal supported weight: {opt_weight:.2f} N")

plt.figure(figsize=(7,5))
plt.plot(y_vals, W_vals/1000, label="Tradeoff Curve (Weight vs Height)")
plt.scatter(opt_height, opt_weight/1000, color='red', zorder=5, 
            label="Optimal Tradeoff Point")
plt.xlabel("Lift Height (cm)")
plt.ylabel("Max Supported Weight (kN)")
plt.title("Optimized Height–Weight Tradeoff (IMA55)")
plt.grid(True)
plt.legend()
plt.show()
