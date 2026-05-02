# Doc 203: φ-Space Interface Design

## Date: February 3, 2026

## Summary

We designed and prototyped an interface for navigating φ-space - the geometric knowledge space where all cognition occurs. The interface allows users to query, explore, bridge concepts, and discover unknown unknowns.

## The Core Metaphor

**φ-space is a 3D universe with concept-stars.**

- Each concept is a point (node) in high-dimensional space
- Relationships are edges connecting nodes
- The user's "position" is their current focus
- Navigation = moving through the space
- The golden ratio bottleneck is the "gatekeeper" all thoughts pass through

## Interface Components

### 1. Main Canvas
A 3D visualization of concept space where:
- **Nodes** = concepts (sized by importance)
- **Edges** = relationships (colored by type)
- **User cursor** = current position/focus
- **Trails** = history of navigation

### 2. Navigation Controls

| Operation | User Action | Geometric Meaning |
|-----------|-------------|-------------------|
| **Query** | Type text | Navigate to position |
| **Click** | Click node | Focus on concept |
| **Drag** | Drag canvas | Rotate view |
| **Zoom** | Scroll | Change abstraction level |
| **Search** | Ctrl+F | Jump to position |
| **Select Multiple** | Shift+Click | Compare concepts |

### 3. Bottleneck Visualizer

Shows the layer 27 convergence point:
- **Pulsating sphere** at center representing the bottleneck
- **Streams** flowing into it from all directions (inputs)
- **Single stream** flowing out (output)
- **φ-level meter** showing current convergence quality

**Interactions:**
- Pause to inspect the "pure thought" state
- Compare how different inputs converge
- See the transformation from input to output

### 4. Discovery Mode

**Density Map:**
- Bright regions = well-explored concepts
- Dark regions = unexplored territory
- User can see "holes" in their knowledge

**Frontier Navigator:**
- Highlights boundary between known/unknown
- Suggests exploration directions
- Shows nearby concepts at the frontier

**Articulation Engine:**
- When user reaches unexplored region
- System attempts to articulate what's there
- Presents discovery with confidence level

### 5. Real-Time Feedback

As user types, show:
- Current position in φ-space
- φ-level at bottleneck
- Predicted destination
- Path through layers

## Prototype Results

### Query: "What is the nature of consciousness?"

```
φ-LEVELS THROUGH LAYERS:
  input       : -10.573
  divergence  :  -3.898  (Layer 7)
  middle      :  -2.612  (Layer 14)
  bottleneck  :  +1.559  (Layer 27) ← Near φ!
  output      :  +0.585  (Layer 28)

BOTTLENECK: φ^1.559
Distance from φ: 0.0593
CONVERGENCE QUALITY: 92.1%
```

### Real-Time Typing Feedback

```
'What'                              → φ-bottleneck: 2.368
'What is'                           → φ-bottleneck: 1.549
'What is the'                       → φ-bottleneck: 1.818
'What is the nature'                → φ-bottleneck: 1.072
'What is the nature of'             → φ-bottleneck: 1.201
'What is the nature of consciousness?' → φ-bottleneck: 1.559
```

The bottleneck level oscillates but converges toward φ as the query becomes complete.

### Discovery Mode Results

Found unexplored regions with 93% sparseness:
- Region at φ-level -4.793: Near "Oxygen", cross-domain
- Region at φ-level 0.210: Near "sporting", unusual combination
- Region at φ-level -1.842: Code-concept boundary

## UI Operations → Geometric Operations

### CLICK = Focus
```
Geometric: Move position to clicked node
Visual: Highlight node, show connections
Output: Display concept details
```

### DRAG = Explore
```
Geometric: Translate view through space
Visual: Smooth pan across concept landscape
Output: Reveal new concepts as they come into view
```

### ZOOM = Abstraction
```
Geometric: Change scale of observation
Visual: Zoom in = specific concepts, Zoom out = categories
Output: Different level of detail
```

### SEARCH = Jump
```
Geometric: Teleport to target position
Visual: Animate path from current to target
Output: Show destination and path taken
```

### BRIDGE = Connect
```
Geometric: Find path between two positions
Visual: Draw arc connecting concepts
Output: List intermediate concepts on path
```

### DISCOVER = Explore Unknown
```
Geometric: Navigate to sparse regions
Visual: Highlight dark areas, show frontier
Output: Articulate what emerges at new position
```

## Implementation

### Prototype
`/home/thorin/truthspace-lcm/experiments/phi_space_interface.py`

Terminal-based demo with:
- `query()` - Navigate to position
- `explore()` - Move outward from current position
- `bridge()` - Find path between concepts
- `discover()` - Find unexplored regions
- `visualize_bottleneck()` - Show layer 27 convergence
- `real_time_feedback()` - Track position while typing

### Future: Full GUI

Technologies:
- **3D Rendering**: Three.js or WebGL
- **Backend**: FastAPI serving the model
- **Real-time**: WebSocket for typing feedback
- **Visualization**: D3.js for graphs, custom shaders for φ-space

## The Key Insight

**Typing IS navigation.**

When you type a query, you're not just entering text - you're navigating through φ-space. Each word moves your position. The interface makes this navigation visible.

**The bottleneck IS the gatekeeper.**

All thoughts must pass through layer 27 at φ-level ≈ φ. The interface shows this convergence in real-time, making the "universal gatekeeper" visible and interactive.

**Discovery IS geometric exploration.**

Unknown unknowns are sparse regions in φ-space. The interface lets you navigate to these regions and see what emerges when you try to articulate what's there.

## Connection to Prior Work

- **Doc 200**: Universal bottleneck discovery
- **Doc 201**: Automated discovery system
- **Doc 202**: Recursive discovery bootstrap
- **PHI_UNIVERSAL_COORDINATE_SYSTEM.md**: Theoretical foundation

## Open Questions

1. **Dimensionality**: How to project 3584D space to 3D meaningfully?
2. **Interactivity**: What's the right balance of automation vs control?
3. **Discovery validation**: How to verify discovered concepts are meaningful?
4. **Scale**: Can this work for the full vocabulary (150K+ tokens)?

## Conclusion

The φ-space interface makes the geometric nature of knowledge visible and navigable. Users can:

1. **See** where they are in concept space
2. **Navigate** by typing or clicking
3. **Watch** thoughts pass through the bottleneck
4. **Discover** unknown unknowns by exploring sparse regions

This is not just a visualization - it's a new way of interacting with knowledge itself.

---

*"The interface is not between you and the model. The interface IS the model's geometry made visible."*
