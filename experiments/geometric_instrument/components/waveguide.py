"""
Module 1: Waveguide — The Residual Stream
==========================================

The medium through which all signals propagate. Carries multiple
independent signals by superposition (addition). Orthogonal signals
do not interfere.

Optical analog: Broadband optical fiber / waveguide
Composition rule: Pure addition (⊕)

Specification:
    Dimensions:     d (typically 3584)
    Mode capacity:  Up to d orthogonal channels
    Orthogonality:  mean |cos(mode_i, mode_j)| < 0.05
    Operation:      Vector addition with residual connections
"""

import numpy as np


class Waveguide:
    """The medium. Carries signals by superposition."""

    def __init__(self, d_model, state=None):
        """Initialize a d-dimensional waveguide.
        
        Args:
            d_model: Dimensionality of the waveguide (e.g. 3584)
            state: Optional initial state vector [d_model]
        """
        self.d_model = d_model
        if state is not None:
            assert state.shape[-1] == d_model, f"State dim {state.shape[-1]} != d_model {d_model}"
            self._state = state.astype(np.float32).copy()
        else:
            self._state = np.zeros(d_model, dtype=np.float32)

    def inject(self, signal):
        """Add a signal to the waveguide (residual connection).
        
        This is the fundamental operation: superposition by addition.
        Multiple injections accumulate without interference as long
        as the signals occupy orthogonal subspaces.
        
        Args:
            signal: numpy array of shape [d_model]
        """
        self._state = self._state + signal.astype(np.float32)

    def read(self):
        """Read the current state of the waveguide.
        
        Returns:
            Copy of the current d-dimensional state vector.
        """
        return self._state.copy()

    def fork(self):
        """Create a copy for branching (e.g., attention + MLP parallel paths).
        
        Returns:
            New Waveguide with the same state.
        """
        return Waveguide(self.d_model, state=self._state)

    def project(self, direction):
        """Project the waveguide state onto a direction.
        
        Args:
            direction: Unit vector [d_model]
            
        Returns:
            Scalar projection (dot product).
        """
        return float(np.dot(self._state, direction))

    def norm(self):
        """Return the L2 norm of the current state."""
        return float(np.linalg.norm(self._state))

    def cosine_to(self, other_state):
        """Cosine similarity between current state and another vector.
        
        Args:
            other_state: numpy array [d_model]
            
        Returns:
            Cosine similarity scalar.
        """
        n1 = np.linalg.norm(self._state)
        n2 = np.linalg.norm(other_state)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(self._state, other_state) / (n1 * n2))

    @staticmethod
    def measure_orthogonality(signals):
        """Measure pairwise orthogonality of a set of signals.
        
        Args:
            signals: list of numpy arrays [d_model]
            
        Returns:
            dict with mean_abs_cosine, max_abs_cosine
        """
        n = len(signals)
        if n < 2:
            return {'mean_abs_cosine': 0.0, 'max_abs_cosine': 0.0}
        
        cosines = []
        for i in range(n):
            ni = np.linalg.norm(signals[i])
            if ni < 1e-12:
                continue
            for j in range(i + 1, n):
                nj = np.linalg.norm(signals[j])
                if nj < 1e-12:
                    continue
                c = abs(float(np.dot(signals[i], signals[j]) / (ni * nj)))
                cosines.append(c)
        
        if not cosines:
            return {'mean_abs_cosine': 0.0, 'max_abs_cosine': 0.0}
        
        return {
            'mean_abs_cosine': float(np.mean(cosines)),
            'max_abs_cosine': float(np.max(cosines)),
        }
