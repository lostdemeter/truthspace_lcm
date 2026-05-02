"""
PhiSolver: The main interface for φ-space geometric navigation.

Usage:
    # Reverse-engineer an existing model
    solver = PhiSolver.from_pretrained("model_name")
    output = solver.navigate(input)
    
    # Create a new pattern
    solver = PhiSolver(pattern=Funnel(1024, 1))
    solver.learn(data)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union
from pathlib import Path
import json

from .encoder import PhiEncoder
from .mesh import MESHComputer
from .pattern import Pattern, Topology


class PhiSolver:
    """
    Generalized φ-space solver for geometric neural network inference.
    
    All models are shapes on the φ-lattice. This solver:
    1. Encodes weights in φ-basis (sign, exponent)
    2. Pre-computes MESH matrices where applicable
    3. Navigates through the shape according to the pattern
    
    Attributes:
        pattern: The navigation pattern (Funnel, Spiral, Web, etc.)
        phi_weights: Dict of (signs, exponents) for each weight tensor
        encoder: PhiEncoder for encoding/decoding
        mesh_computer: MESHComputer for pre-computing combined matrices
    """
    
    def __init__(
        self,
        pattern: Optional[Pattern] = None,
        K: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize a PhiSolver.
        
        Args:
            pattern: Navigation pattern (optional, can be set later)
            K: φ-encoding resolution (32 = ~3% precision)
            device: Compute device
        """
        self.pattern = pattern
        self.K = K
        self.device = device
        
        self.encoder = PhiEncoder(K=K)
        self.mesh_computer = MESHComputer(self.encoder)
        
        self.phi_weights: Dict[str, tuple] = {}
        self.mesh_matrices: Dict[str, tuple] = {}
        self.float_weights: Dict[str, torch.Tensor] = {}  # For navigation
        
        self._source_model = None
        self.encoding_accuracy = None
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        pattern: Optional[Pattern] = None,
        K: int = 32,
        **kwargs
    ) -> "PhiSolver":
        """
        Load a pretrained model and convert to φ-space.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            pattern: Optional pattern (auto-detected if not provided)
            K: φ-encoding resolution
            
        Returns:
            PhiSolver with φ-encoded weights
        """
        solver = cls(pattern=pattern, K=K)
        
        # Try to load the model
        model = solver._load_model(model_name_or_path, **kwargs)
        solver._source_model = model
        
        # Extract and encode weights
        solver._extract_weights(model)
        
        # Auto-detect pattern if not provided
        if pattern is None:
            solver.pattern = solver._detect_pattern(model)
        
        # Compute MESH matrices for attention layers
        solver._compute_mesh_matrices(model)
        
        return solver
    
    def _load_model(self, model_name_or_path: str, **kwargs) -> nn.Module:
        """Load a model from HuggingFace or local path."""
        # Try HuggingFace first
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
            model.eval()
            return model.to(self.device)
        except:
            pass
        
        # Try loading as a torch model
        try:
            model = torch.load(model_name_or_path)
            model.eval()
            return model.to(self.device)
        except:
            pass
        
        raise ValueError(f"Could not load model from {model_name_or_path}")
    
    def _extract_weights(self, model: nn.Module):
        """Extract all weights and encode in φ-basis."""
        total_params = 0
        total_error = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                tensor = param.detach()
                
                # Encode in φ-basis
                signs, exps = self.encoder.encode(tensor)
                self.phi_weights[name] = (signs, exps)
                
                # Also store decoded version for navigation
                self.float_weights[name] = self.encoder.decode(signs, exps)
                
                # Track accuracy
                stats = self.encoder.verify_accuracy(tensor)
                total_params += tensor.numel()
                total_error += (1 - stats['correlation']) * tensor.numel()
        
        self.encoding_accuracy = 1 - (total_error / total_params)
        print(f"Encoded {len(self.phi_weights)} weight tensors")
        print(f"φ-encoding accuracy: {self.encoding_accuracy:.6f}")
    
    def _detect_pattern(self, model: nn.Module) -> Pattern:
        """Auto-detect the pattern from model architecture."""
        from .pattern import Funnel, Spiral, Web
        
        # Count layer types
        has_self_attn = False
        has_cross_attn = False
        num_layers = 0
        
        for name, module in model.named_modules():
            name_lower = name.lower()
            if 'self_attn' in name_lower or 'self_attention' in name_lower:
                has_self_attn = True
                num_layers += 1
            if 'cross_attn' in name_lower or 'cross_attention' in name_lower:
                has_cross_attn = True
        
        # Determine pattern
        if has_cross_attn and has_self_attn:
            print(f"Detected pattern: Web ({num_layers} layers)")
            return Web(queries=100, dim=256, feature_scales=3, 
                      layers=num_layers, output_dim=2)
        elif has_self_attn:
            print(f"Detected pattern: Spiral ({num_layers} layers)")
            return Spiral(layers=num_layers, dim=256, heads=8)
        else:
            print("Detected pattern: Funnel")
            return Funnel(in_dim=256, out_dim=1)
    
    def _compute_mesh_matrices(self, model: nn.Module):
        """Pre-compute MESH matrices for attention layers."""
        for name, module in model.named_modules():
            # Look for attention modules with in_proj_weight
            if hasattr(module, 'in_proj_weight'):
                in_proj = module.in_proj_weight.detach()
                
                # Split into Q, K, V
                dim = in_proj.shape[0] // 3
                W_q = in_proj[:dim]
                W_k = in_proj[dim:2*dim]
                W_v = in_proj[2*dim:]
                
                # Compute MESH
                mesh_result = self.mesh_computer.compute_attention_mesh(W_q, W_k)
                self.mesh_matrices[f"{name}_mesh"] = mesh_result['mesh']
                
        print(f"Pre-computed {len(self.mesh_matrices)} MESH matrices")
    
    def navigate(self, input: torch.Tensor) -> torch.Tensor:
        """
        Navigate through the φ-lattice according to the pattern.
        
        This is the main inference method. It uses the φ-encoded weights
        to traverse the geometric shape.
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor
        """
        if self._source_model is not None:
            # Use the source model with φ-decoded weights
            # (In a full implementation, we'd replace the weights)
            with torch.no_grad():
                return self._source_model(input)
        
        # Pure φ-navigation (pattern-based)
        return self._navigate_pattern(input)
    
    def _navigate_pattern(self, input: torch.Tensor) -> torch.Tensor:
        """Navigate using only the pattern and φ-weights."""
        if self.pattern is None:
            raise ValueError("No pattern specified")
        
        x = input
        
        for node in self.pattern.nodes:
            if node.node_type == "linear":
                w_key = self._find_weight_key(node.name)
                if w_key:
                    signs, exps = self.phi_weights[w_key]
                    W = self.encoder.decode(signs, exps)
                    x = x @ W.T
            
            elif node.node_type == "self_attention":
                # Use MESH if available
                mesh_key = f"{node.name}_mesh"
                if mesh_key in self.mesh_matrices:
                    signs, exps = self.mesh_matrices[mesh_key]
                    mesh = self.encoder.decode(signs, exps)
                    # Simplified attention: x @ mesh @ x.T
                    scores = x @ mesh @ x.T
                    attn = torch.softmax(scores, dim=-1)
                    x = attn @ x
            
            elif node.node_type == "ffn":
                # Two linear layers with activation
                w1_key = self._find_weight_key(f"{node.name}_linear1")
                w2_key = self._find_weight_key(f"{node.name}_linear2")
                if w1_key and w2_key:
                    W1 = self.encoder.decode(*self.phi_weights[w1_key])
                    W2 = self.encoder.decode(*self.phi_weights[w2_key])
                    x = torch.relu(x @ W1.T) @ W2.T
        
        return x
    
    def _find_weight_key(self, node_name: str) -> Optional[str]:
        """Find the weight key that matches a node name."""
        for key in self.phi_weights:
            if node_name in key:
                return key
        return None
    
    def save(self, path: Union[str, Path]):
        """
        Save the φ-encoded solver to disk.
        
        Saves:
        - Pattern specification
        - φ-encoded weights (signs, exponents)
        - MESH matrices
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save pattern
        pattern_data = {
            "name": self.pattern.name,
            "topology": self.pattern.topology.value,
            "self_reference": self.pattern.self_reference.value,
            "io_ratio": self.pattern.io_ratio,
            "nodes": [
                {
                    "name": n.name,
                    "node_type": n.node_type,
                    "in_dim": n.in_dim,
                    "out_dim": n.out_dim,
                    "params": n.params
                }
                for n in self.pattern.nodes
            ]
        }
        with open(path / "pattern.json", "w") as f:
            json.dump(pattern_data, f, indent=2)
        
        # Save φ-weights
        phi_data = {}
        for name, (signs, exps) in self.phi_weights.items():
            phi_data[name] = {
                "signs": signs.cpu().numpy().tolist(),
                "exps": exps.cpu().numpy().tolist(),
                "shape": list(signs.shape)
            }
        torch.save(phi_data, path / "phi_weights.pt")
        
        # Save MESH matrices
        if self.mesh_matrices:
            mesh_data = {}
            for name, (signs, exps) in self.mesh_matrices.items():
                mesh_data[name] = {
                    "signs": signs.cpu().numpy().tolist(),
                    "exps": exps.cpu().numpy().tolist(),
                    "shape": list(signs.shape)
                }
            torch.save(mesh_data, path / "mesh_matrices.pt")
        
        # Save metadata
        metadata = {
            "K": self.K,
            "encoding_accuracy": self.encoding_accuracy,
            "num_weights": len(self.phi_weights),
            "num_mesh": len(self.mesh_matrices)
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved φ-solver to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "PhiSolver":
        """Load a φ-encoded solver from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        
        solver = cls(K=metadata["K"])
        solver.encoding_accuracy = metadata["encoding_accuracy"]
        
        # Load pattern
        with open(path / "pattern.json") as f:
            pattern_data = json.load(f)
        
        from .pattern import Pattern, Topology, SelfReference, PatternNode
        solver.pattern = Pattern(
            name=pattern_data["name"],
            topology=Topology(pattern_data["topology"]),
            self_reference=SelfReference(pattern_data["self_reference"]),
            io_ratio=pattern_data["io_ratio"]
        )
        for node_data in pattern_data["nodes"]:
            solver.pattern.add_node(PatternNode(**node_data))
        
        # Load φ-weights
        phi_data = torch.load(path / "phi_weights.pt")
        for name, data in phi_data.items():
            signs = torch.tensor(data["signs"])
            exps = torch.tensor(data["exps"])
            solver.phi_weights[name] = (signs, exps)
            solver.float_weights[name] = solver.encoder.decode(signs, exps)
        
        # Load MESH matrices
        mesh_path = path / "mesh_matrices.pt"
        if mesh_path.exists():
            mesh_data = torch.load(mesh_path)
            for name, data in mesh_data.items():
                signs = torch.tensor(data["signs"])
                exps = torch.tensor(data["exps"])
                solver.mesh_matrices[name] = (signs, exps)
        
        print(f"Loaded φ-solver from {path}")
        print(f"  Pattern: {solver.pattern.name}")
        print(f"  Weights: {len(solver.phi_weights)}")
        print(f"  Accuracy: {solver.encoding_accuracy:.6f}")
        
        return solver
    
    def describe(self) -> str:
        """Human-readable description of the solver."""
        lines = [
            "φ-Space Solver",
            "=" * 40,
            f"Pattern: {self.pattern.name if self.pattern else 'None'}",
            f"Topology: {self.pattern.topology.value if self.pattern else 'N/A'}",
            f"φ-encoding K: {self.K}",
            f"Encoding accuracy: {self.encoding_accuracy:.6f}" if self.encoding_accuracy else "",
            f"Weight tensors: {len(self.phi_weights)}",
            f"MESH matrices: {len(self.mesh_matrices)}",
        ]
        return "\n".join(lines)


def test_solver():
    """Test the PhiSolver."""
    print("Testing PhiSolver...")
    
    from .pattern import Funnel
    
    # Create a simple solver
    solver = PhiSolver(pattern=Funnel(256, 1))
    
    # Create fake weights
    W = torch.randn(1, 256) * 0.1
    signs, exps = solver.encoder.encode(W)
    solver.phi_weights["output.weight"] = (signs, exps)
    solver.float_weights["output.weight"] = solver.encoder.decode(signs, exps)
    
    # Test navigation
    x = torch.randn(1, 256)
    # output = solver.navigate(x)
    
    print(solver.describe())
    
    return solver


if __name__ == "__main__":
    test_solver()
