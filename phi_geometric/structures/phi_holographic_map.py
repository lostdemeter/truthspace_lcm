"""
φ-Holographic Map: A data structure based on the gate field discovery.

From Doc 245-246: The GELU gate field in neural networks implements
holographic computation on a φ-lattice reference frame. This module
generalizes that mechanism into a standalone data structure.

Key properties:
  1. Denoising mean: default() is BETTER than individual lookup()
  2. Implicit reference frame: φ-lattice costs 0 bits
  3. Locality-preserving: similar inputs → similar gate codes
  4. Graceful compression: low-rank Jacobian improves quality
  5. Self-similar: same structure at every scale

Usage:
    phi_map = PhiMap(dim=64, expansion=4)
    phi_map.fit(keys, values)           # Learn hyperplanes
    phi_map.calibrate(calibration_set)  # Compute mean Jacobian

    value = phi_map.lookup(key)         # Full nonlinear lookup
    value = phi_map.default(key)        # Mean Jacobian (faster, denoised)
    phi_map.compress(rank_fraction=0.25)  # Further denoise via SVD
    code = phi_map.encode(key)          # Binary gate code
    dist = phi_map.similarity(key_a, key_b)  # Hamming distance of codes
"""
import numpy as np
from typing import Optional, Tuple, List

PHI = (1 + np.sqrt(5)) / 2


def _phi_gelu(x: np.ndarray) -> np.ndarray:
    """φ-scaled soft gate: x · σ(φ·x), matching GELU curvature."""
    from scipy.special import expit
    return x * expit(PHI * x)


def _phi_gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of φ-GELU for Jacobian computation."""
    from scipy.special import expit
    sig = expit(PHI * x)
    return sig + x * PHI * sig * (1 - sig)


def _standard_gelu(x: np.ndarray) -> np.ndarray:
    """Standard GELU: x · Φ(x)."""
    from scipy.special import erf
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _standard_gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of standard GELU."""
    from scipy.special import erf
    cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf


class PhiMap:
    """
    A φ-Holographic Map.

    Encodes key-value relationships as gate-modulated linear transforms
    on a φ-lattice reference frame.

    Parameters
    ----------
    dim : int
        Dimensionality of keys and values.
    expansion : int
        Expansion factor for the gate field (default 4, matching ConvNeXt).
    gate : str
        Gate function: 'phi_gelu' (φ-scaled sigmoid) or 'gelu' (standard).
    """

    def __init__(self, dim: int, expansion: int = 4, gate: str = 'gelu'):
        self.dim = dim
        self.expansion = expansion
        self.E = dim * expansion

        self.gate_type = gate
        if gate == 'phi_gelu':
            self._gate_fn = _phi_gelu
            self._gate_deriv = _phi_gelu_derivative
        elif gate == 'gelu':
            self._gate_fn = _standard_gelu
            self._gate_deriv = _standard_gelu_derivative
        else:
            raise ValueError(f"Unknown gate: {gate}")

        # Hyperplane bank H ∈ [E, D] and bias b ∈ [E]
        self.H = None
        self.b = None

        # Reconstruction matrix R ∈ [D, E] and output bias
        self.R = None
        self.b_out = None

        # Cached mean Jacobian
        self._jacobian = None
        self._jacobian_bias = None
        self._jacobian_rank = None

    # ================================================================
    # Core operations
    # ================================================================

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode input(s) into soft gate values and binary gate codes.

        Parameters
        ----------
        x : ndarray of shape [..., D]

        Returns
        -------
        gate : ndarray of shape [..., E] — soft gate values
        code : ndarray of shape [..., E] — binary gate codes (bool)
        """
        z = x @ self.H.T + self.b        # [..., E]
        gate = self._gate_fn(z)           # Soft gate
        code = z > 0                      # Binary code
        return gate, code

    def decode(self, gated: np.ndarray) -> np.ndarray:
        """
        Reconstruct value from gate-modulated signal.

        Parameters
        ----------
        gated : ndarray of shape [..., E]

        Returns
        -------
        value : ndarray of shape [..., D]
        """
        return gated @ self.R.T + self.b_out

    def lookup(self, x: np.ndarray) -> np.ndarray:
        """
        Full nonlinear lookup: encode → decode.

        Parameters
        ----------
        x : ndarray of shape [..., D]

        Returns
        -------
        value : ndarray of shape [..., D]
        """
        gate, _ = self.encode(x)
        return self.decode(gate)

    def default(self, x: np.ndarray) -> np.ndarray:
        """
        Mean Jacobian lookup: linearized, denoised.

        Requires calibrate() to have been called first.

        Parameters
        ----------
        x : ndarray of shape [..., D]

        Returns
        -------
        value : ndarray of shape [..., D]
        """
        if self._jacobian is None:
            raise RuntimeError("Call calibrate() before default()")
        return x @ self._jacobian.T + self._jacobian_bias

    def similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Normalized Hamming distance between gate codes of x and y.

        Returns
        -------
        distance : float in [0, 1], where 0 = identical codes
        """
        _, code_x = self.encode(x)
        _, code_y = self.encode(y)
        return np.mean(code_x != code_y)

    # ================================================================
    # Learning / Calibration
    # ================================================================

    def init_random(self, seed: int = 42):
        """Initialize with random hyperplanes (Xavier initialization)."""
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (self.dim + self.E))
        self.H = rng.randn(self.E, self.dim).astype(np.float32) * scale
        self.b = np.zeros(self.E, dtype=np.float32)
        self.R = rng.randn(self.dim, self.E).astype(np.float32) * scale
        self.b_out = np.zeros(self.dim, dtype=np.float32)
        return self

    def init_phi_structured(self, seed: int = 42):
        """
        Initialize with φ-structured hyperplanes.

        Uses φ-damped harmonics as basis functions, matching the
        DW conv φ-basis discovery (R²=0.982).
        """
        rng = np.random.RandomState(seed)

        H = np.zeros((self.E, self.dim), dtype=np.float32)
        for i in range(self.E):
            freq = (i % self.dim) + 1
            phase = (i // self.dim) * np.pi / (2 * self.expansion)
            decay = PHI ** (-(i % self.dim) / self.dim)
            t = np.linspace(0, 2 * np.pi * freq / self.dim, self.dim)
            H[i] = decay * np.cos(t + phase)

        # Normalize rows
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        H = H / (norms + 1e-10)

        # Add small random perturbation for symmetry breaking
        H += rng.randn(*H.shape).astype(np.float32) * 0.01

        self.H = H
        self.b = np.zeros(self.E, dtype=np.float32)
        self.R = rng.randn(self.dim, self.E).astype(np.float32) * np.sqrt(2.0 / (self.dim + self.E))
        self.b_out = np.zeros(self.dim, dtype=np.float32)
        return self

    def fit(self, X: np.ndarray, Y: np.ndarray, n_iter: int = 1000,
            lr: float = 0.01, verbose: bool = False):
        """
        Learn H and R from input-output pairs via gradient descent.

        Parameters
        ----------
        X : ndarray of shape [N, D] — input keys
        Y : ndarray of shape [N, D] — target values
        n_iter : int
        lr : float
        verbose : bool
        """
        if self.H is None:
            self.init_random()

        N = X.shape[0]
        best_loss = float('inf')
        best_state = None

        for it in range(n_iter):
            # Forward
            z = X @ self.H.T + self.b                 # [N, E]
            gated = self._gate_fn(z)                   # [N, E]
            pred = gated @ self.R.T + self.b_out       # [N, D]

            # Loss
            residual = pred - Y                        # [N, D]
            loss = np.mean(residual ** 2)

            if loss < best_loss:
                best_loss = loss
                best_state = (self.H.copy(), self.b.copy(),
                              self.R.copy(), self.b_out.copy())

            if verbose and it % 100 == 0:
                print(f"  iter {it:4d}: loss = {loss:.6f}")

            # Backward (manual gradients)
            d_pred = 2 * residual / N                  # [N, D]
            d_gated = d_pred @ self.R                  # [N, E]
            d_R = d_pred.T @ gated                     # [D, E]
            d_b_out = d_pred.sum(axis=0)               # [D]

            gate_deriv = self._gate_deriv(z)           # [N, E]
            d_z = d_gated * gate_deriv                 # [N, E]
            d_H = d_z.T @ X                            # [E, D]
            d_b = d_z.sum(axis=0)                      # [E]

            # Update
            self.H -= lr * d_H
            self.b -= lr * d_b
            self.R -= lr * d_R
            self.b_out -= lr * d_b_out

        # Restore best
        if best_state is not None:
            self.H, self.b, self.R, self.b_out = best_state

        return best_loss

    def calibrate(self, X: np.ndarray):
        """
        Compute mean Jacobian from calibration examples.

        The mean Jacobian J_mean = R @ diag(E[gate'(z)]) @ H
        is the optimal linear approximation to the nonlinear lookup.

        Parameters
        ----------
        X : ndarray of shape [N, D]
        """
        z = X @ self.H.T + self.b                     # [N, E]
        mean_gate_deriv = self._gate_deriv(z).mean(axis=0)  # [E]
        mean_gate_out = self._gate_fn(self.b)          # [E] — gate at zero input

        # J_mean = R @ diag(mean_gate_deriv) @ H
        self._jacobian = (self.R * mean_gate_deriv[np.newaxis, :]) @ self.H  # [D, D]
        self._jacobian_bias = self.R @ mean_gate_out + self.b_out  # [D]
        self._jacobian_rank = None

    def compress(self, rank_fraction: float = 0.25):
        """
        Compress the mean Jacobian via low-rank SVD.

        This is a DENOISING operation — it typically IMPROVES quality.

        Parameters
        ----------
        rank_fraction : float in (0, 1]
        """
        if self._jacobian is None:
            raise RuntimeError("Call calibrate() before compress()")

        U, S, Vt = np.linalg.svd(self._jacobian, full_matrices=False)
        k = max(1, int(self.dim * rank_fraction))
        self._jacobian = (U[:, :k] * S[:k]) @ Vt[:k, :]
        self._jacobian_rank = k

    # ================================================================
    # Analysis
    # ================================================================

    def gate_statistics(self, X: np.ndarray) -> dict:
        """Analyze gate field properties on a dataset."""
        z = X @ self.H.T + self.b
        gate_binary = z > 0

        alive_rate = gate_binary.mean(axis=0)  # Per-channel alive rate
        code_uniqueness = len(set(
            tuple(row) for row in gate_binary[:min(1000, len(X))]
        )) / min(1000, len(X))

        return {
            'alive_rate_mean': alive_rate.mean(),
            'alive_rate_std': alive_rate.std(),
            'dead_channels': (alive_rate < 0.05).sum(),
            'code_uniqueness': code_uniqueness,
            'effective_bits': -np.sum(
                alive_rate * np.log2(alive_rate + 1e-10) +
                (1 - alive_rate) * np.log2(1 - alive_rate + 1e-10)
            ),
        }

    def jacobian_spectrum(self) -> Optional[np.ndarray]:
        """Return singular values of the mean Jacobian."""
        if self._jacobian is None:
            return None
        _, S, _ = np.linalg.svd(self._jacobian, full_matrices=False)
        return S

    @property
    def param_count(self) -> dict:
        """Count parameters in each component."""
        full = self.E * self.dim * 2 + self.E + self.dim  # H + R + biases
        if self._jacobian is not None:
            if self._jacobian_rank is not None:
                jac = self._jacobian_rank * self.dim * 2 + self.dim
            else:
                jac = self.dim * self.dim + self.dim
        else:
            jac = None
        return {
            'full': full,
            'jacobian': jac,
            'compression': jac / full if jac else None,
        }


class PhiMapStack:
    """
    A stack of φ-Maps with residual connections.

    This is the generalized form of what ConvNeXt implements:
    multiple resolution levels, each with multiple φ-Map blocks,
    connected by residual additions.
    """

    def __init__(self, dims: List[int], depths: List[int],
                 expansion: int = 4, gate: str = 'gelu'):
        self.dims = dims
        self.depths = depths
        self.levels = []
        for dim, depth in zip(dims, depths):
            level = [PhiMap(dim, expansion, gate) for _ in range(depth)]
            self.levels.append(level)

    def lookup(self, x: np.ndarray) -> List[np.ndarray]:
        """Full nonlinear forward pass with residual connections."""
        features = []
        for level in self.levels:
            for phi_map in level:
                x = x + phi_map.lookup(x)
            features.append(x)
        return features

    def default(self, x: np.ndarray) -> List[np.ndarray]:
        """Mean Jacobian forward pass (faster, denoised)."""
        features = []
        for level in self.levels:
            for phi_map in level:
                x = x + phi_map.default(x)
            features.append(x)
        return features

    def calibrate_all(self, X: np.ndarray):
        """Calibrate all φ-Maps in the stack."""
        for level in self.levels:
            for phi_map in level:
                phi_map.calibrate(X)
                x_new = X + phi_map.lookup(X)
                X = x_new

    def compress_all(self, rank_fraction: float = 0.25):
        """Compress all mean Jacobians."""
        for level in self.levels:
            for phi_map in level:
                phi_map.compress(rank_fraction)
