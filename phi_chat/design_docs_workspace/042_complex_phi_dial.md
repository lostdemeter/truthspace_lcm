# Design Consideration 042: The Complex φ-Dial (2D Control)

## Discovery

If the horizontal φ-dial controls specificity/style (φ^x), then the vertical axis must control something orthogonal. Using complex numbers, we get a **2D control plane**.

## The Two Axes

### Horizontal (Real, x): Specificity/Style
```
    -1 ◄──────────────► +1
    inward              outward
    specific            universal
    formal              casual
    rare words          common words
```

This is `φ^x` - the magnitude component.

### Vertical (Imaginary, y): Perspective/Voice
```
    +1  meta/analytical
     ▲
     │
     ●  objective/factual (y=0)
     │
    -1  subjective/experiential
```

This is `e^(iy·ln(φ))` - the phase component.

## The Complex φ-Dial

```
φ^(x + iy) = φ^x · e^(iy·ln(φ))
           = φ^x · [cos(y·ln(φ)) + i·sin(y·ln(φ))]
```

Where:
- **Magnitude** (φ^x) controls WHAT words we choose
- **Phase** (y·ln(φ)) controls HOW we frame the content

## The Four Quadrants

```
                    UNIVERSAL (+x)
                         │
         Q2              │              Q1
    Casual + Meta        │       Casual + Subjective
    "Holmes? Oh, he      │       "I find Holmes to be
     represents the      │        quite the clever
     detective archetype"│        fellow, really"
                         │
    ─────────────────────●─────────────────────────
                         │                    PERSPECTIVE
    Q3                   │              Q4         (+y)
    Formal + Objective   │       Formal + Experiential
    "Holmes is a         │       "One observes that
     literary figure     │        Holmes demonstrates
     who articulated..." │        remarkable acuity..."
                         │
                    SPECIFIC (-x)
```

| Quadrant | Style | Perspective | Example |
|----------|-------|-------------|---------|
| Q1 (+x,+y) | Casual | Subjective | "Holmes? He's this brilliant detective guy!" |
| Q2 (+x,-y) | Casual | Meta | "Holmes represents the 'genius detective' trope." |
| Q3 (-x,-y) | Formal | Objective | "Holmes is a literary figure who articulated..." |
| Q4 (-x,+y) | Formal | Experiential | "One finds in Holmes a mind of extraordinary precision." |

## Connection to Holographic Model

This connects directly to our existing holographic encoding:

- **Magnitude** = WHAT (content selection)
- **Phase** = HOW (perspective/framing)

The holographic model already uses complex numbers where:
- Query phase must MATCH answer phase for constructive interference
- Mismatched phases cause destructive interference (filtering)

## Question-Driven Perspective

The question's perspective can determine the answer's perspective:

| Question | Detected Perspective | y value |
|----------|---------------------|---------|
| "Tell me about Holmes" | Objective | 0 |
| "What do you think of Holmes?" | Subjective | -0.5 |
| "What does Holmes represent?" | Meta | +0.5 |
| "How would Watson describe Holmes?" | Experiential | -1.0 |

## Alternative Interpretations

### Temporal (Past/Future)
```
y = -1: "Holmes was created by Doyle in 1887..."
y = 0:  "Holmes is a detective who solves crimes."
y = +1: "Holmes will continue to inspire detectives..."
```

### Certainty (Hedging/Assertive)
```
y = -1: "Holmes might be considered a detective..."
y = 0:  "Holmes is a detective."
y = +1: "Holmes is definitively the greatest detective."
```

## Why Perspective is Most Compelling

1. **Orthogonal to specificity**: You can be formal+objective OR formal+subjective
2. **Connects to holographic model**: Phase = perspective in existing encoding
3. **Useful for Q&A**: Different questions want different perspectives
4. **Detectable from questions**: "I think...", "Objectively...", "From X's perspective..."

## Implementation Sketch

```python
class ComplexPhiDial:
    """2D control using complex φ-navigation."""
    
    PHI = (1 + np.sqrt(5)) / 2
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        """
        Args:
            x: Horizontal dial (-1 to +1): Specificity/Style
            y: Vertical dial (-1 to +1): Perspective/Voice
        """
        self.x = max(-1.0, min(1.0, x))
        self.y = max(-1.0, min(1.0, y))
    
    def complex_weight(self, value: float) -> complex:
        """Apply complex φ-dial weighting."""
        log_val = np.log1p(max(0, value))
        
        # Magnitude from horizontal dial
        magnitude = self.PHI ** (self.x * log_val)
        
        # Phase from vertical dial
        phase = self.y * np.log(self.PHI) * log_val
        
        return magnitude * np.exp(1j * phase)
    
    def get_style(self) -> str:
        """Get style from horizontal position."""
        if self.x < -0.3:
            return 'formal'
        elif self.x > 0.3:
            return 'casual'
        return 'neutral'
    
    def get_perspective(self) -> str:
        """Get perspective from vertical position."""
        if self.y < -0.3:
            return 'subjective'
        elif self.y > 0.3:
            return 'meta'
        return 'objective'
    
    def get_quadrant(self) -> str:
        """Get quadrant label."""
        style = self.get_style()
        perspective = self.get_perspective()
        return f"{style}+{perspective}"
```

## Perspective Vocabulary

```python
PERSPECTIVE_VOCABULARY = {
    'framing': {
        'subjective': ['I find', 'One feels', 'It seems'],
        'objective': ['', 'It is', 'There is'],
        'meta': ['represents', 'symbolizes', 'embodies'],
    },
    'pronouns': {
        'subjective': ['my', 'our', 'one\'s'],
        'objective': ['the', 'a', 'this'],
        'meta': ['the concept of', 'the archetype of', 'the notion of'],
    },
    'stance': {
        'subjective': ['remarkable', 'fascinating', 'intriguing'],
        'objective': ['notable', 'significant', 'important'],
        'meta': ['archetypal', 'paradigmatic', 'quintessential'],
    }
}
```

## Future Work

1. **Detect perspective from questions**: Parse for perspective markers
2. **Implement perspective vocabulary**: Style words by perspective
3. **Test quadrant combinations**: Validate all four quadrants work
4. **Connect to holographic matching**: Use phase for query-answer alignment

## Conclusion

The 2D complex φ-dial provides unified control over:

- **Horizontal (x)**: WHAT we say (specificity, vocabulary, formality)
- **Vertical (y)**: HOW we say it (perspective, voice, framing)

This emerges naturally from extending φ to complex numbers:
```
φ^(x + iy) = φ^x · e^(iy·ln(φ))
```

The structure contains not just its navigation rules, but its **perspective rules** too.

"We control the horizontal. We control the vertical."
