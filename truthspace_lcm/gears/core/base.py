"""
Base classes for the Gear system.

This module provides the foundational classes that all gears inherit from:
- Quaternion: 4D rotation encoding for gear parameters
- GearState: The state object passed between gears
- Gear: Abstract base class for all gears
- GearChain: Container for composing gears

Author: Lesley Gushurst
License: GPLv3
"""

import math
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Quaternion:
    """
    Quaternion for encoding gear parameters.
    
    Components:
    - w: scalar (often represents strength/confidence)
    - x: first vector component
    - y: second vector component
    - z: third vector component
    
    Quaternions can be multiplied to chain transformations.
    """
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication (Hamilton product)."""
        return Quaternion(
            w=self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            x=self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            y=self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            z=self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w,
        )
    
    def conjugate(self) -> 'Quaternion':
        """Return the conjugate (inverse rotation)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self) -> float:
        """Return the norm (magnitude)."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Quaternion':
        """Return a normalized quaternion."""
        n = self.norm()
        if n < 1e-10:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def scale(self, factor: float) -> 'Quaternion':
        """Scale the quaternion by a factor."""
        return Quaternion(self.w * factor, self.x * factor, 
                         self.y * factor, self.z * factor)
    
    @staticmethod
    def slerp(q1: 'Quaternion', q2: 'Quaternion', t: float) -> 'Quaternion':
        """Spherical linear interpolation between two quaternions."""
        q1 = q1.normalize()
        q2 = q2.normalize()
        
        dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
        
        if dot < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot
        
        if dot > 0.9995:
            result = Quaternion(
                q1.w + t*(q2.w - q1.w),
                q1.x + t*(q2.x - q1.x),
                q1.y + t*(q2.y - q1.y),
                q1.z + t*(q2.z - q1.z),
            )
            return result.normalize()
        
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        
        s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        
        return Quaternion(
            s1*q1.w + s2*q2.w,
            s1*q1.x + s2*q2.x,
            s1*q1.y + s2*q2.y,
            s1*q1.z + s2*q2.z,
        )


@dataclass
class GearState:
    """
    State object passed between gears in the chain.
    
    This contains all the information that gears can read and modify.
    Each gear transforms the state and passes it to the next gear.
    """
    # Core content
    entity: str = ""
    role: str = "entity"
    actions: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    
    # Accumulated quaternion from gear chain
    accumulated_q: Quaternion = field(default_factory=Quaternion)
    
    # Style flags
    use_prefix: bool = False
    use_gerunds: bool = True
    connector: str = "that involves"
    target_connector: str = "particularly"
    
    # Signal-specific
    signal_prefix: str = ""
    signal_suffix: str = ""
    signal_style: str = "default"
    
    # Tense
    tense: str = "present"
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def clone(self) -> 'GearState':
        """Create a deep copy of the state."""
        import copy
        return copy.deepcopy(self)


class Gear(ABC):
    """
    Abstract base class for all gears.
    
    A gear is a transformation unit that:
    1. Takes a GearState as input
    2. Applies some transformation
    3. Returns the modified GearState
    
    Gears can be composed in chains to create complex transformations.
    
    Attributes:
        name: Human-readable name for the gear
        ratio: Control parameter (0.0 to 1.0) that affects transformation strength
        quaternion: 4D parameter encoding for the gear
        enabled: Whether the gear is active
    """
    
    def __init__(self, name: str, ratio: float = 1.0):
        self.name = name
        self.ratio = ratio
        self.quaternion = Quaternion(1, 0, 0, 0)
        self.enabled = True
    
    @abstractmethod
    def forward(self, state: GearState) -> GearState:
        """
        Apply the gear's transformation to the state.
        
        Args:
            state: The current gear state
            
        Returns:
            The transformed gear state
        """
        pass
    
    def backward(self, state: GearState) -> GearState:
        """
        Apply the inverse transformation (for bidirectional gears).
        
        Default implementation returns state unchanged.
        Override in subclasses that support backward transformation.
        """
        return state
    
    def set_ratio(self, ratio: float) -> 'Gear':
        """Set the gear ratio and return self for chaining."""
        self.ratio = max(0.0, min(1.0, ratio))
        return self
    
    def set_quaternion(self, q: Quaternion) -> 'Gear':
        """Set the gear's quaternion and return self for chaining."""
        self.quaternion = q
        return self
    
    def enable(self) -> 'Gear':
        """Enable the gear."""
        self.enabled = True
        return self
    
    def disable(self) -> 'Gear':
        """Disable the gear."""
        self.enabled = False
        return self
    
    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.name}(ratio={self.ratio:.2f}, {status})"


class GearChain:
    """
    A chain of gears that transforms state sequentially.
    
    Gears are applied in order, with each gear receiving the output
    of the previous gear as its input.
    
    Usage:
        chain = GearChain()
        chain.add(RoleGear())
        chain.add(ActionGear())
        chain.add(OutputGear())
        
        result = chain.process(initial_state)
    """
    
    def __init__(self, name: str = "GearChain"):
        self.name = name
        self.gears: List[Gear] = []
    
    def add(self, gear: Gear, position: int = -1) -> 'GearChain':
        """
        Add a gear to the chain.
        
        Args:
            gear: The gear to add
            position: Position to insert (-1 = end)
            
        Returns:
            Self for chaining
        """
        if position == -1:
            self.gears.append(gear)
        else:
            self.gears.insert(position, gear)
        return self
    
    def remove(self, name: str) -> 'GearChain':
        """Remove a gear by name."""
        self.gears = [g for g in self.gears if g.name != name]
        return self
    
    def get(self, name: str) -> Optional[Gear]:
        """Get a gear by name."""
        for gear in self.gears:
            if gear.name == name:
                return gear
        return None
    
    def process(self, state: GearState) -> Any:
        """
        Process state through all gears in the chain.
        
        Args:
            state: Initial gear state
            
        Returns:
            Final output (string if last gear is OutputGear, else GearState)
        """
        current = state
        
        for gear in self.gears:
            if gear.enabled:
                current = gear.forward(current)
        
        return current
    
    def process_backward(self, state: GearState) -> GearState:
        """Process state backward through the chain (for correction propagation)."""
        current = state
        
        for gear in reversed(self.gears):
            if gear.enabled:
                current = gear.backward(current)
        
        return current
    
    def set_ratio(self, name: str, ratio: float) -> 'GearChain':
        """Set the ratio for a specific gear."""
        gear = self.get(name)
        if gear:
            gear.set_ratio(ratio)
        return self
    
    def enable_all(self) -> 'GearChain':
        """Enable all gears."""
        for gear in self.gears:
            gear.enable()
        return self
    
    def disable(self, name: str) -> 'GearChain':
        """Disable a specific gear."""
        gear = self.get(name)
        if gear:
            gear.disable()
        return self
    
    def __repr__(self) -> str:
        gear_names = [g.name for g in self.gears if g.enabled]
        return f"{self.name}: {' → '.join(gear_names)}"
    
    def __len__(self) -> int:
        return len(self.gears)
    
    def __iter__(self):
        return iter(self.gears)
