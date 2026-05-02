#!/usr/bin/env python3
"""
φ-Decoder AIG Synthesis
========================

Actually synthesize the φ-decoder as an AIG and optimize it.

This creates a complete, working AIG representation that could be:
1. Simulated in Python
2. Exported to BLIF for ABC optimization
3. Synthesized to Verilog for FPGA/ASIC

Key insight: The φ-basis transformation means we just need to SUM
the transformed dimensions - no learned weights needed!

In φ-basis: depth = Σ φ_dim_i (just addition!)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class AIGWire:
    """A wire in the AIG (can be inverted)."""
    node_id: int
    inverted: bool = False
    
    def __neg__(self):
        """Invert the wire."""
        return AIGWire(self.node_id, not self.inverted)
    
    def __repr__(self):
        return f"{'~' if self.inverted else ''}{self.node_id}"


@dataclass 
class AIGGate:
    """An AND gate in the AIG."""
    output_id: int
    input_a: AIGWire
    input_b: AIGWire


class AIG:
    """
    And-Inverter Graph implementation.
    
    Supports:
    - Primary inputs (PI)
    - AND gates
    - Primary outputs (PO)
    - Simulation
    - BLIF export
    """
    
    def __init__(self, name: str = "circuit"):
        self.name = name
        self.next_id = 1  # 0 is reserved for constant FALSE
        
        self.inputs: Dict[str, List[AIGWire]] = {}  # name -> [wire per bit]
        self.outputs: Dict[str, List[AIGWire]] = {}  # name -> [wire per bit]
        self.gates: List[AIGGate] = []
        self.wire_names: Dict[int, str] = {0: "FALSE"}
    
    def constant(self, value: bool) -> AIGWire:
        """Return constant TRUE or FALSE."""
        if value:
            return AIGWire(0, inverted=True)  # ~FALSE = TRUE
        return AIGWire(0, inverted=False)  # FALSE
    
    def add_input(self, name: str, bits: int = 1) -> List[AIGWire]:
        """Add a primary input (possibly multi-bit)."""
        wires = []
        for i in range(bits):
            wire = AIGWire(self.next_id)
            self.wire_names[self.next_id] = f"{name}[{i}]" if bits > 1 else name
            self.next_id += 1
            wires.append(wire)
        self.inputs[name] = wires
        return wires
    
    def add_and(self, a: AIGWire, b: AIGWire) -> AIGWire:
        """Add an AND gate."""
        output = AIGWire(self.next_id)
        self.gates.append(AIGGate(self.next_id, a, b))
        self.next_id += 1
        return output
    
    def add_or(self, a: AIGWire, b: AIGWire) -> AIGWire:
        """OR using De Morgan: a OR b = ~(~a AND ~b)."""
        return -self.add_and(-a, -b)
    
    def add_xor(self, a: AIGWire, b: AIGWire) -> AIGWire:
        """XOR: a XOR b = (a AND ~b) OR (~a AND b)."""
        and1 = self.add_and(a, -b)
        and2 = self.add_and(-a, b)
        return self.add_or(and1, and2)
    
    def add_mux(self, sel: AIGWire, a: AIGWire, b: AIGWire) -> AIGWire:
        """MUX: sel ? a : b = (sel AND a) OR (~sel AND b)."""
        and1 = self.add_and(sel, a)
        and2 = self.add_and(-sel, b)
        return self.add_or(and1, and2)
    
    def add_half_adder(self, a: AIGWire, b: AIGWire) -> Tuple[AIGWire, AIGWire]:
        """Half adder: returns (sum, carry)."""
        sum_out = self.add_xor(a, b)
        carry = self.add_and(a, b)
        return sum_out, carry
    
    def add_full_adder(self, a: AIGWire, b: AIGWire, cin: AIGWire) -> Tuple[AIGWire, AIGWire]:
        """Full adder: returns (sum, carry_out)."""
        # sum = a XOR b XOR cin
        # cout = (a AND b) OR (cin AND (a XOR b))
        ab_xor = self.add_xor(a, b)
        sum_out = self.add_xor(ab_xor, cin)
        
        ab_and = self.add_and(a, b)
        cin_ab = self.add_and(cin, ab_xor)
        carry = self.add_or(ab_and, cin_ab)
        
        return sum_out, carry
    
    def add_ripple_adder(self, a: List[AIGWire], b: List[AIGWire], 
                         cin: Optional[AIGWire] = None) -> Tuple[List[AIGWire], AIGWire]:
        """
        Ripple-carry adder for multi-bit addition.
        
        Returns (sum_bits, carry_out).
        """
        if len(a) != len(b):
            raise ValueError("Inputs must have same width")
        
        if cin is None:
            cin = self.constant(False)
        
        sum_bits = []
        carry = cin
        
        for i in range(len(a)):
            s, carry = self.add_full_adder(a[i], b[i], carry)
            sum_bits.append(s)
        
        return sum_bits, carry
    
    def add_output(self, name: str, wires: List[AIGWire]):
        """Add primary outputs."""
        self.outputs[name] = wires
    
    def gate_count(self) -> int:
        """Return number of AND gates."""
        return len(self.gates)
    
    def simulate(self, input_values: Dict[str, int]) -> Dict[str, int]:
        """
        Simulate the AIG with given input values.
        
        Args:
            input_values: Dict mapping input names to integer values
            
        Returns:
            Dict mapping output names to integer values
        """
        # Initialize wire values
        wire_values = {0: False}  # Constant FALSE
        
        # Set input values
        for name, wires in self.inputs.items():
            value = input_values.get(name, 0)
            for i, wire in enumerate(wires):
                wire_values[wire.node_id] = bool((value >> i) & 1)
        
        # Evaluate gates in order
        for gate in self.gates:
            a_val = wire_values[gate.input_a.node_id] ^ gate.input_a.inverted
            b_val = wire_values[gate.input_b.node_id] ^ gate.input_b.inverted
            wire_values[gate.output_id] = a_val and b_val
        
        # Read outputs
        results = {}
        for name, wires in self.outputs.items():
            value = 0
            for i, wire in enumerate(wires):
                bit = wire_values[wire.node_id] ^ wire.inverted
                if bit:
                    value |= (1 << i)
            results[name] = value
        
        return results
    
    def to_blif(self) -> str:
        """Export to BLIF format for ABC optimization."""
        lines = [f".model {self.name}"]
        
        # Inputs
        input_names = []
        for name, wires in self.inputs.items():
            for i, wire in enumerate(wires):
                input_names.append(self.wire_names[wire.node_id])
        lines.append(f".inputs {' '.join(input_names)}")
        
        # Outputs
        output_names = []
        for name, wires in self.outputs.items():
            for i, wire in enumerate(wires):
                out_name = f"{name}[{i}]" if len(wires) > 1 else name
                output_names.append(out_name)
        lines.append(f".outputs {' '.join(output_names)}")
        
        # Constant FALSE
        lines.append(".names const0")
        lines.append("")  # Empty truth table = always 0
        
        # Gates
        for gate in self.gates:
            a_name = self.wire_names.get(gate.input_a.node_id, f"n{gate.input_a.node_id}")
            b_name = self.wire_names.get(gate.input_b.node_id, f"n{gate.input_b.node_id}")
            out_name = f"n{gate.output_id}"
            self.wire_names[gate.output_id] = out_name
            
            lines.append(f".names {a_name} {b_name} {out_name}")
            # Truth table for AND (with possible inversions)
            a_val = '0' if gate.input_a.inverted else '1'
            b_val = '0' if gate.input_b.inverted else '1'
            lines.append(f"{a_val}{b_val} 1")
        
        # Output connections
        for name, wires in self.outputs.items():
            for i, wire in enumerate(wires):
                out_name = f"{name}[{i}]" if len(wires) > 1 else name
                src_name = self.wire_names.get(wire.node_id, f"n{wire.node_id}")
                if wire.inverted:
                    lines.append(f".names {src_name} {out_name}")
                    lines.append("0 1")  # Inverter
                else:
                    lines.append(f".names {src_name} {out_name}")
                    lines.append("1 1")  # Buffer
        
        lines.append(".end")
        return "\n".join(lines)


def build_phi_multiplier(aig: AIG, x: List[AIGWire], phi_exp: float) -> List[AIGWire]:
    """
    Build a circuit to multiply x by φ^exp using shift-add.
    
    Uses Fibonacci approximation: φ^n ≈ F(k)/F(j) for appropriate k,j.
    """
    # Fibonacci-based approximations for common exponents
    # These are chosen for good accuracy with small denominators
    approx_table = {
        -2.0: (144, 377),   # φ^-2 ≈ 144/377 ≈ 0.382
        -1.0: (233, 377),   # φ^-1 ≈ 233/377 ≈ 0.618
        0.0: (1, 1),        # φ^0 = 1
        1.0: (377, 233),    # φ^1 ≈ 377/233 ≈ 1.618
        2.0: (377, 144),    # φ^2 ≈ 377/144 ≈ 2.618
    }
    
    # For half-integer exponents, use sqrt approximations
    # φ^0.5 ≈ 1.272 ≈ 5/4 (simple) or 89/70 (better)
    # φ^-0.5 ≈ 0.786 ≈ 11/14 
    half_approx = {
        -1.5: (70, 144),    # φ^-1.5 ≈ 0.486
        -0.5: (11, 14),     # φ^-0.5 ≈ 0.786
        0.5: (89, 70),      # φ^0.5 ≈ 1.271
        1.5: (144, 70),     # φ^1.5 ≈ 2.058
    }
    
    if phi_exp in approx_table:
        num, den = approx_table[phi_exp]
    elif phi_exp in half_approx:
        num, den = half_approx[phi_exp]
    else:
        # Default to closest integer exponent
        closest = round(phi_exp)
        num, den = approx_table.get(closest, (1, 1))
    
    # Multiply by numerator using shift-add
    result = multiply_by_constant(aig, x, num)
    
    # Division by denominator would require more complex logic
    # For now, we'll handle this by scaling outputs appropriately
    # In practice, we'd use fixed-point arithmetic
    
    return result


def multiply_by_constant(aig: AIG, x: List[AIGWire], constant: int) -> List[AIGWire]:
    """
    Multiply x by a constant using shift-add.
    
    constant is decomposed into sum of powers of 2.
    """
    if constant == 0:
        return [aig.constant(False)] * len(x)
    
    if constant == 1:
        return x.copy()
    
    # Find which bits are set in constant
    shifts = []
    c = constant
    bit = 0
    while c > 0:
        if c & 1:
            shifts.append(bit)
        c >>= 1
        bit += 1
    
    # Create shifted versions and add them
    # Result width = input width + highest shift
    result_width = len(x) + max(shifts)
    
    # Start with first shifted version
    first_shift = shifts[0]
    result = [aig.constant(False)] * first_shift + x + [aig.constant(False)] * (result_width - len(x) - first_shift)
    
    # Add remaining shifted versions
    for shift in shifts[1:]:
        shifted = [aig.constant(False)] * shift + x + [aig.constant(False)] * (result_width - len(x) - shift)
        result, _ = aig.add_ripple_adder(result, shifted)
    
    return result


def build_phi_decoder_aig(n_dims: int = 8, bit_width: int = 8) -> AIG:
    """
    Build complete φ-decoder as an AIG.
    
    In φ-basis, the decoder is just: depth = Σ φ_dim_i
    
    Each φ_dim_i is already transformed, so we just need to sum!
    """
    aig = AIG("phi_decoder")
    
    # Add inputs for each dimension
    dim_inputs = []
    for i in range(n_dims):
        dim_wires = aig.add_input(f"dim{i}", bit_width)
        dim_inputs.append(dim_wires)
    
    # Sum all dimensions using a tree of adders
    # This is more efficient than a chain
    current_level = dim_inputs
    result_width = bit_width
    
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                # Pad to same width
                a = current_level[i] + [aig.constant(False)] * (result_width + 1 - len(current_level[i]))
                b = current_level[i+1] + [aig.constant(False)] * (result_width + 1 - len(current_level[i+1]))
                sum_bits, carry = aig.add_ripple_adder(a[:result_width+1], b[:result_width+1])
                next_level.append(sum_bits + [carry])
            else:
                # Odd one out, pass through
                next_level.append(current_level[i])
        current_level = next_level
        result_width += 1
    
    # Output is the final sum
    aig.add_output("depth", current_level[0][:bit_width + 4])  # Keep reasonable width
    
    return aig


def test_phi_decoder_aig():
    """Test the φ-decoder AIG."""
    print("=" * 70)
    print("φ-DECODER AIG TEST")
    print("=" * 70)
    print()
    
    # Build small decoder for testing
    n_dims = 4
    bit_width = 8
    
    print(f"Building φ-decoder AIG ({n_dims} dims, {bit_width}-bit)...")
    aig = build_phi_decoder_aig(n_dims, bit_width)
    
    print(f"  AND gates: {aig.gate_count()}")
    print(f"  Inputs: {sum(len(w) for w in aig.inputs.values())}")
    print(f"  Outputs: {sum(len(w) for w in aig.outputs.values())}")
    print()
    
    # Test simulation
    print("Testing simulation...")
    test_cases = [
        {f"dim{i}": 10 for i in range(n_dims)},  # All 10s -> sum = 40
        {f"dim{i}": i * 5 for i in range(n_dims)},  # 0, 5, 10, 15 -> sum = 30
        {f"dim{i}": 255 if i == 0 else 0 for i in range(n_dims)},  # Just dim0 = 255
    ]
    
    for inputs in test_cases:
        result = aig.simulate(inputs)
        expected = sum(inputs.values())
        actual = result['depth']
        status = "✓" if actual == expected else "✗"
        print(f"  {status} Inputs: {list(inputs.values())} -> depth={actual} (expected {expected})")
    
    print()
    return aig


def analyze_scaling():
    """Analyze how AIG complexity scales with decoder size."""
    print("=" * 70)
    print("AIG SCALING ANALYSIS")
    print("=" * 70)
    print()
    
    print(f"{'Dims':>6} {'Bits':>6} {'Gates':>10} {'Gates/Dim':>12}")
    print("-" * 40)
    
    for n_dims in [4, 8, 16, 32, 50]:
        for bit_width in [8, 16]:
            aig = build_phi_decoder_aig(n_dims, bit_width)
            gates = aig.gate_count()
            per_dim = gates / n_dims
            print(f"{n_dims:>6} {bit_width:>6} {gates:>10,} {per_dim:>12.1f}")
    
    print()


def main():
    """Run AIG synthesis experiments."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " φ-DECODER AIG SYNTHESIS ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Test the decoder
    aig = test_phi_decoder_aig()
    
    # Analyze scaling
    analyze_scaling()
    
    # Export BLIF
    print("=" * 70)
    print("BLIF EXPORT")
    print("=" * 70)
    print()
    
    blif = aig.to_blif()
    blif_path = Path(__file__).parent / "phi_decoder.blif"
    blif_path.write_text(blif)
    print(f"Exported to: {blif_path}")
    print(f"BLIF size: {len(blif)} bytes")
    print()
    print("To optimize with ABC:")
    print(f"  abc> read_blif {blif_path}")
    print("  abc> strash; balance; rewrite; refactor")
    print("  abc> print_stats")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    # Full decoder estimate
    full_aig = build_phi_decoder_aig(50, 8)
    print(f"Full φ-decoder (50 dims, 8-bit):")
    print(f"  AND gates: {full_aig.gate_count():,}")
    print(f"  Estimated transistors: {full_aig.gate_count() * 6:,}")
    print()
    print("Key insight: In φ-basis, the decoder is JUST ADDITION!")
    print("No multiplications needed - the φ-transform is pre-applied to data.")
    print()


if __name__ == "__main__":
    main()
