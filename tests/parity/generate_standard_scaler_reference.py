"""Generate reference data for StandardScaler parity tests.

Produces CSV files that the C++ parity tests read to verify
bit-level consistency with scikit-learn.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

# Reference dataset
X = np.array([
    [1.0, -1.0,  2.0],
    [2.0,  0.0,  0.0],
    [0.0,  1.0, -1.0],
    [1.0,  1.0,  1.0],
    [3.0, -1.0,  0.0],
], dtype=np.float64)

scaler = StandardScaler()
Z = scaler.fit_transform(X)
X_inv = scaler.inverse_transform(Z)

np.savetxt(os.path.join(out_dir, "X.csv"), X, delimiter=",", fmt="%.18e")
np.savetxt(os.path.join(out_dir, "Z.csv"), Z, delimiter=",", fmt="%.18e")
np.savetxt(os.path.join(out_dir, "X_inv.csv"), X_inv, delimiter=",", fmt="%.18e")
np.savetxt(os.path.join(out_dir, "mean.csv"), scaler.mean_.reshape(1, -1),
           delimiter=",", fmt="%.18e")
np.savetxt(os.path.join(out_dir, "var.csv"), scaler.var_.reshape(1, -1),
           delimiter=",", fmt="%.18e")
np.savetxt(os.path.join(out_dir, "scale.csv"), scaler.scale_.reshape(1, -1),
           delimiter=",", fmt="%.18e")

print("Reference data written to", out_dir)
print(f"  X:     {X.shape}")
print(f"  Z:     {Z.shape}")
print(f"  mean:  {scaler.mean_}")
print(f"  var:   {scaler.var_}")
print(f"  scale: {scaler.scale_}")
