"""
φ-Space Solver Library

A generalized library for geometric neural network inference.
All models are shapes on the φ-lattice, and inference is navigation.

Usage:
    from phi_solver import PhiSolver
    from phi_solver.patterns import Funnel, Spiral, Web
    
    # Reverse-engineer an existing model
    solver = PhiSolver.from_pretrained("model_name")
    output = solver.navigate(input)
    
    # Or create a new pattern
    solver = PhiSolver(pattern=Funnel(in_dim=1024, out_dim=1))
    solver.learn(data)
"""

from .encoder import PhiEncoder
from .mesh import MESHComputer
from .solver import PhiSolver
from .pattern import Pattern

__version__ = "0.1.0"
__all__ = ["PhiEncoder", "MESHComputer", "PhiSolver", "Pattern"]
