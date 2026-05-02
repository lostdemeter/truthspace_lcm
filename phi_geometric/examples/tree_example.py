"""
Tree Pattern Example: Multi-Task Learning

The Tree pattern is divergent (one → many).
Used for: Multi-task learning, ensemble outputs, universal scene understanding.

Characteristics:
    - Single shared encoder
    - Multiple output branches
    - Each branch specializes in one task

This example builds a multi-task model without training.

Author: TruthSpace LCM Project
Date: February 5, 2026
"""

import torch
from typing import Dict, List, Optional

from ..core import (
    GeometricAI, ProblemSpec, IOSpec, DataType,
    PhiEncoder, Tree
)


class TreeMultiTask:
    """
    A multi-task model using the Tree pattern.
    
    The Tree pattern diverges from a shared representation
    to multiple specialized outputs. Each branch handles
    one task.
    
    Example:
        model = TreeMultiTask(
            input_dim=256,
            tasks={
                "depth": 1,
                "normals": 3,
                "edges": 1,
                "segmentation": 20
            }
        )
        
        # Inject task knowledge
        model.inject_task_knowledge("depth", "Measures distance from camera")
        model.inject_task_knowledge("normals", "Surface orientation vectors")
        
        # Run all tasks
        outputs = model.forward(features)
        # outputs = {"depth": ..., "normals": ..., ...}
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        tasks: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the multi-task model.
        
        Args:
            input_dim: Dimension of shared features
            tasks: Dict of {task_name: output_dim}
        """
        self.input_dim = input_dim
        self.tasks = tasks or {
            "depth": 1,
            "normals": 3,
            "edges": 1
        }
        
        # Create separate GeometricAI for each task
        # (In practice, would share encoder)
        self.task_models: Dict[str, GeometricAI] = {}
        
        for task_name, output_dim in self.tasks.items():
            problem = ProblemSpec(
                name=f"tree_{task_name}",
                inputs=[IOSpec("features", DataType.VECTOR, (input_dim,), "shared features")],
                outputs=[IOSpec(task_name, DataType.VECTOR, (output_dim,), task_name)],
            )
            self.task_models[task_name] = GeometricAI(problem)
        
        # Inject default knowledge
        self._inject_default_knowledge()
    
    def _inject_default_knowledge(self):
        """Inject default multi-task knowledge."""
        for task_name, model in self.task_models.items():
            model.inject_knowledge(f"This branch predicts {task_name}")
            model.inject_knowledge("Tasks share underlying features")
    
    def inject_task_knowledge(self, task_name: str, fact: str):
        """
        Inject knowledge for a specific task.
        
        Args:
            task_name: Name of the task
            fact: Knowledge to inject
        """
        if task_name in self.task_models:
            self.task_models[task_name].inject_knowledge(fact)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run all tasks on shared features.
        
        Args:
            features: Shared features [input_dim] or [B, input_dim]
            
        Returns:
            Dict of {task_name: output}
        """
        outputs = {}
        for task_name, model in self.task_models.items():
            outputs[task_name] = model(features)
        return outputs
    
    def forward_task(self, features: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        Run a single task.
        
        Args:
            features: Shared features
            task_name: Which task to run
            
        Returns:
            Task output
        """
        if task_name not in self.task_models:
            raise ValueError(f"Unknown task: {task_name}")
        return self.task_models[task_name](features)
    
    def stats(self) -> Dict[str, Dict]:
        """Get statistics for all tasks."""
        return {name: model.stats() for name, model in self.task_models.items()}


def demo_tree_multitask():
    """Demonstrate the Tree multi-task model."""
    print("=" * 70)
    print("TREE PATTERN EXAMPLE: Multi-Task Learning")
    print("=" * 70)
    
    # Create multi-task model
    model = TreeMultiTask(
        input_dim=64,
        tasks={
            "depth": 1,
            "normals": 3,
            "edges": 1,
            "segmentation": 10
        }
    )
    
    # Inject task-specific knowledge
    model.inject_task_knowledge("depth", "Depth is inversely related to apparent size")
    model.inject_task_knowledge("normals", "Normals are unit vectors perpendicular to surface")
    model.inject_task_knowledge("edges", "Edges occur at depth discontinuities")
    model.inject_task_knowledge("segmentation", "Segments group similar regions")
    
    print("\nMulti-Task Model created:")
    print(f"  Input dim: {model.input_dim}")
    print(f"  Tasks: {list(model.tasks.keys())}")
    print(f"  Pattern: Tree (divergent)")
    
    # Test forward
    print("\n--- Multi-Task Inference ---")
    features = torch.randn(64)
    outputs = model.forward(features)
    
    for task_name, output in outputs.items():
        print(f"  {task_name}: shape = {output.shape}, range = [{output.min():.3f}, {output.max():.3f}]")
    
    # Single task
    print("\n--- Single Task Inference ---")
    depth = model.forward_task(features, "depth")
    print(f"  Depth only: {depth.item():.4f}")
    
    # Stats
    print("\n--- Statistics ---")
    stats = model.stats()
    for task_name, task_stats in stats.items():
        print(f"  {task_name}: pattern={task_stats['pattern']}, memory={task_stats['memory_size']}")
    
    print("\n" + "=" * 70)
    print("TREE EXAMPLE COMPLETE")
    print("=" * 70)
    
    return model


if __name__ == "__main__":
    demo_tree_multitask()
