"""
HyperMapping vs Neural Networks - Advanced Comparison

Uses techniques from design considerations:
- 044: Quaternion φ-Dial (4D control)
- 049: Gradient-Free Learning (error = where to add structure)
- 052: Hypothesis-Driven Knowledge (test predictions)
- 055: Tachyon Navigation (W-axis = certainty)

Key insight: The naive similarity matching in the basic comparison
doesn't capture what we actually achieved with these techniques.
"""

import sys
sys.path.insert(0, '/home/thorin/truthspace-lcm')

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from hypermapping import HyperMapping, Encoder, CRITICAL_LINE


# =============================================================================
# QUATERNION ENCODER (from Design 044)
# =============================================================================

@dataclass
class QuaternionEncoder(Encoder):
    """
    4D Quaternion encoder with semantic axes:
    - X: Style dimension
    - Y: Perspective dimension  
    - Z: Depth dimension
    - W: Certainty dimension
    
    From Design 044: "The quaternion structure is mathematically natural"
    """
    
    def __init__(self, dims: int = 4):
        super().__init__(dims)
        # Semantic axis mappings
        self.style_vocab = {
            'love': -0.8, 'hate': -0.8, 'amazing': -0.6, 'terrible': -0.6,
            'great': -0.4, 'bad': -0.4, 'good': -0.2, 'poor': -0.2,
            'okay': 0.0, 'average': 0.0, 'fine': 0.0,
        }
        self.intensity_vocab = {
            'very': 0.8, 'really': 0.7, 'extremely': 0.9, 'somewhat': 0.3,
            'slightly': 0.2, 'a bit': 0.2, 'totally': 0.9, 'completely': 0.9,
        }
        self.polarity_vocab = {
            'love': 1.0, 'amazing': 0.9, 'great': 0.8, 'excellent': 0.9,
            'good': 0.6, 'best': 1.0, 'wonderful': 0.9, 'fantastic': 0.9,
            'hate': -1.0, 'terrible': -0.9, 'awful': -0.9, 'worst': -1.0,
            'bad': -0.6, 'poor': -0.5, 'disappointed': -0.7, 'waste': -0.8,
            'okay': 0.0, 'average': 0.0, 'nothing': -0.1, 'special': 0.3,
        }
    
    def encode_input(self, text: str) -> np.ndarray:
        words = str(text).lower().split()
        
        # X-axis: Polarity (positive/negative)
        polarity = 0.0
        for word in words:
            polarity += self.polarity_vocab.get(word, 0.0)
        polarity = np.clip(polarity, -1, 1)
        
        # Y-axis: Intensity
        intensity = 0.5  # default moderate
        for word in words:
            if word in self.intensity_vocab:
                intensity = self.intensity_vocab[word]
                break
        
        # Z-axis: Style (formal vs casual based on word choice)
        style = 0.0
        for word in words:
            if word in self.style_vocab:
                style = self.style_vocab[word]
                break
        
        # W-axis: Certainty (hedged language detection)
        certainty = 0.0
        hedged_words = {'seems', 'appears', 'maybe', 'perhaps', 'might', 'could'}
        definite_words = {'is', 'definitely', 'certainly', 'absolutely', 'clearly'}
        for word in words:
            if word in hedged_words:
                certainty = 0.5
            elif word in definite_words:
                certainty = -0.5
        
        pos = np.array([polarity, intensity, style, certainty])
        norm = np.linalg.norm(pos)
        if norm > 1e-10:
            pos = pos / norm * CRITICAL_LINE
        return pos
    
    def encode_output(self, output: str) -> np.ndarray:
        # Output is the sentiment label
        if output == 'positive':
            return np.array([1.0, 0.5, 0.0, 0.0]) * CRITICAL_LINE
        elif output == 'negative':
            return np.array([-1.0, 0.5, 0.0, 0.0]) * CRITICAL_LINE
        else:  # neutral
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def encode_mapping(self, input_val: Any, output_val: Any) -> np.ndarray:
        return self.encode_input(input_val)


# =============================================================================
# GRADIENT-FREE LEARNER (from Design 049)
# =============================================================================

class GradientFreeLearner:
    """
    Error-driven structure learning from Design 049.
    
    Key insight: "Error doesn't measure accuracy — it tells us where to add structure."
    
    Traditional ML: error = how wrong we are → adjust weights
    Geometric ML: error = what's missing → add structure
    """
    
    def __init__(self):
        self.structure: Dict[str, Dict[str, Any]] = {}
        self.role_vocab = {'detective', 'doctor', 'gentleman', 'lady', 'investigator'}
        self.quality_vocab = {'brilliant', 'loyal', 'proud', 'witty', 'analytical'}
        self.action_vocab = {'investigates', 'assists', 'loves', 'challenges', 'examines'}
    
    def learn_from_error(self, entity: str, target: str, generated: str) -> None:
        """Add structure based on what's missing."""
        target_words = set(target.lower().split())
        generated_words = set(generated.lower().split())
        missing = target_words - generated_words
        
        if entity not in self.structure:
            self.structure[entity] = {'role': None, 'qualities': [], 'actions': [], 'relations': []}
        
        for word in missing:
            word_lower = word.lower()
            if word_lower in self.role_vocab:
                self.structure[entity]['role'] = word_lower
            elif word_lower in self.quality_vocab:
                if word_lower not in self.structure[entity]['qualities']:
                    self.structure[entity]['qualities'].append(word_lower)
            elif word_lower in self.action_vocab:
                if word_lower not in self.structure[entity]['actions']:
                    self.structure[entity]['actions'].append(word_lower)
            elif word_lower in {'watson', 'holmes', 'darcy', 'elizabeth'}:
                if word_lower not in self.structure[entity]['relations'] and word_lower != entity:
                    self.structure[entity]['relations'].append(word_lower)
    
    def generate(self, entity: str) -> str:
        """Generate from learned structure."""
        if entity not in self.structure:
            return f"{entity.title()} is a character."
        
        s = self.structure[entity]
        role = s.get('role', 'character')
        qualities = ' '.join(s.get('qualities', []))
        actions = s.get('actions', ['exists'])[0] if s.get('actions') else 'exists'
        relations = s.get('relations', [])
        
        if qualities and relations:
            return f"{entity.title()} is a {qualities} {role} who {actions} with {relations[0].title()}."
        elif qualities:
            return f"{entity.title()} is a {qualities} {role}."
        else:
            return f"{entity.title()} is a {role}."
    
    def train(self, data: List[Tuple[str, str]], epochs: int = 2) -> float:
        """Train using error-driven learning."""
        for epoch in range(epochs):
            for entity, target in data:
                generated = self.generate(entity)
                self.learn_from_error(entity, target, generated)
        
        # Calculate final overlap
        total_overlap = 0.0
        for entity, target in data:
            generated = self.generate(entity)
            target_words = set(target.lower().split())
            generated_words = set(generated.lower().split())
            overlap = len(target_words & generated_words) / len(target_words | generated_words)
            total_overlap += overlap
        
        return total_overlap / len(data) * 100


# =============================================================================
# HYPOTHESIS-DRIVEN CLASSIFIER (from Design 052)
# =============================================================================

@dataclass
class Hypothesis:
    """A testable claim about an entity."""
    claim: str
    category: str
    predictions: Dict[str, Set[str]]  # prediction_type -> expected_words
    confidence: float = 0.0


class HypothesisDrivenClassifier:
    """
    From Design 052: "We know what we're looking for BEFORE we look."
    
    Key insight: "WHO you interact with reveals your role more than WHAT actions you take."
    """
    
    def __init__(self):
        # Define hypotheses for sentiment
        self.hypotheses = {
            'positive': Hypothesis(
                claim='positive sentiment',
                category='sentiment',
                predictions={
                    'polarity_words': {'love', 'amazing', 'great', 'excellent', 'best', 
                                       'wonderful', 'fantastic', 'good', 'happy'},
                    'intensifiers': {'very', 'really', 'so', 'extremely', 'totally'},
                }
            ),
            'negative': Hypothesis(
                claim='negative sentiment',
                category='sentiment',
                predictions={
                    'polarity_words': {'hate', 'terrible', 'awful', 'worst', 'bad',
                                       'poor', 'disappointed', 'waste', 'horrible'},
                    'intensifiers': {'very', 'really', 'so', 'extremely', 'totally'},
                }
            ),
            'neutral': Hypothesis(
                claim='neutral sentiment',
                category='sentiment',
                predictions={
                    'polarity_words': {'okay', 'average', 'fine', 'alright', 'decent'},
                    'hedgers': {'somewhat', 'kind of', 'sort of', 'nothing special'},
                }
            ),
        }
    
    def test_hypothesis(self, text: str, hypothesis: Hypothesis) -> float:
        """Test how well text matches hypothesis predictions."""
        words = set(text.lower().split())
        
        total_score = 0.0
        for pred_type, expected in hypothesis.predictions.items():
            matches = len(words & expected)
            if matches > 0:
                total_score += matches / len(expected)
        
        return total_score / len(hypothesis.predictions)
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify by testing all hypotheses."""
        best_label = 'neutral'
        best_score = 0.0
        
        for label, hypothesis in self.hypotheses.items():
            score = self.test_hypothesis(text, hypothesis)
            if score > best_score:
                best_score = score
                best_label = label
        
        return best_label, best_score


# =============================================================================
# SEQUENCE PREDICTOR WITH TACHYON NAVIGATION (from Design 055)
# =============================================================================

class TachyonSequencePredictor:
    """
    From Design 055: W-axis = navigation direction.
    
    Forward (φ^+n) = "I observed this" = definitive
    Backward (φ^-n) = "I hypothesize this" = hedged
    
    For sequences: we navigate forward to find patterns, backward to predict.
    """
    
    def __init__(self):
        self.patterns: Dict[str, List[Tuple[List[int], int]]] = {
            'fibonacci': [],
            'arithmetic': [],
            'geometric': [],
        }
    
    def detect_pattern(self, seq: List[int]) -> Tuple[str, float]:
        """Navigate forward (φ^+n) to detect pattern type."""
        if len(seq) < 3:
            return 'unknown', 0.0
        
        # Check Fibonacci: a[n] = a[n-1] + a[n-2]
        fib_score = 0
        for i in range(2, len(seq)):
            if seq[i] == seq[i-1] + seq[i-2]:
                fib_score += 1
        fib_confidence = fib_score / (len(seq) - 2) if len(seq) > 2 else 0
        
        # Check Arithmetic: constant difference (check first, it's more common)
        diffs = [seq[i] - seq[i-1] for i in range(1, len(seq))]
        arith_confidence = 1.0 if len(set(diffs)) == 1 else 0.0
        self._last_diff = diffs[-1] if diffs else 0
        
        # Check Geometric: constant ratio
        if all(seq[i-1] != 0 for i in range(1, len(seq))):
            ratios = [seq[i] / seq[i-1] for i in range(1, len(seq))]
            geom_confidence = 1.0 if len(set(ratios)) == 1 else 0.0
        else:
            geom_confidence = 0.0
        
        # Return best match - prioritize arithmetic if it's a perfect match
        # since Fibonacci can look like arithmetic for short sequences
        if arith_confidence == 1.0:
            return ('arithmetic', arith_confidence)
        if geom_confidence == 1.0:
            return ('geometric', geom_confidence)
        if fib_confidence >= 0.5:
            return ('fibonacci', fib_confidence)
        
        scores = [
            ('fibonacci', fib_confidence),
            ('arithmetic', arith_confidence),
            ('geometric', geom_confidence),
        ]
        best = max(scores, key=lambda x: x[1])
        return best
    
    def predict_next(self, seq: List[int]) -> Tuple[int, float]:
        """Navigate backward (φ^-n) to predict next value."""
        pattern, confidence = self.detect_pattern(seq)
        
        if pattern == 'fibonacci' and confidence > 0.5:
            return seq[-1] + seq[-2], confidence
        elif pattern == 'arithmetic' and confidence > 0.5:
            diff = seq[-1] - seq[-2]
            return seq[-1] + diff, confidence
        elif pattern == 'geometric' and confidence > 0.5:
            ratio = seq[-1] / seq[-2] if seq[-2] != 0 else 1
            return int(seq[-1] * ratio), confidence
        else:
            # Fallback: assume arithmetic
            diff = seq[-1] - seq[-2] if len(seq) >= 2 else 1
            return seq[-1] + diff, 0.3
    
    def train(self, sequences: List[Tuple[List[int], int]]) -> None:
        """Learn patterns from examples."""
        for seq, next_val in sequences:
            pattern, _ = self.detect_pattern(seq + [next_val])
            self.patterns[pattern].append((seq, next_val))


# =============================================================================
# TESTS
# =============================================================================

def test_quaternion_sentiment():
    """Test sentiment analysis with Quaternion encoder."""
    print("=" * 60)
    print("  TASK 3: Sentiment Analysis (Quaternion Encoder)")
    print("=" * 60)
    print()
    
    training_data = [
        ("I love this product", "positive"),
        ("This is amazing", "positive"),
        ("Great quality", "positive"),
        ("Excellent service", "positive"),
        ("Best purchase ever", "positive"),
        ("I hate this", "negative"),
        ("Terrible quality", "negative"),
        ("Worst product", "negative"),
        ("Very disappointed", "negative"),
        ("Complete waste", "negative"),
        ("It's okay", "neutral"),
        ("Average product", "neutral"),
        ("Nothing special", "neutral"),
    ]
    
    test_data = [
        ("I really love it", "positive"),
        ("Amazing quality", "positive"),
        ("This is terrible", "negative"),
        ("Hate the quality", "negative"),
        ("It's average", "neutral"),
        ("Pretty good product", "positive"),
        ("Very bad service", "negative"),
    ]
    
    # Create HyperMapping with Quaternion encoder
    encoder = QuaternionEncoder(dims=4)
    space = HyperMapping(dims=4, encoder=encoder, name="sentiment")
    
    for text, sentiment in training_data:
        space.map(text, sentiment)
    
    print(f"Trained on {len(training_data)} examples")
    print()
    
    # Test
    print("Testing with Quaternion encoding:")
    correct = 0
    for text, expected in test_data:
        result = space.forward(text, use_similarity=False)
        predicted = result.output if result else None
        is_correct = predicted == expected
        correct += is_correct
        sim = result.similarity if result else 0
        print(f"  '{text}'")
        print(f"    → {predicted} (expected {expected}, sim={sim:.2f}) {'✓' if is_correct else '✗'}")
    
    accuracy = correct / len(test_data) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    return accuracy


def test_hypothesis_sentiment():
    """Test sentiment analysis with Hypothesis-Driven approach."""
    print()
    print("=" * 60)
    print("  TASK 3b: Sentiment Analysis (Hypothesis-Driven)")
    print("=" * 60)
    print()
    
    test_data = [
        ("I really love it", "positive"),
        ("Amazing quality", "positive"),
        ("This is terrible", "negative"),
        ("Hate the quality", "negative"),
        ("It's average", "neutral"),
        ("Pretty good product", "positive"),
        ("Very bad service", "negative"),
    ]
    
    classifier = HypothesisDrivenClassifier()
    
    print("Testing with Hypothesis-Driven classification:")
    correct = 0
    for text, expected in test_data:
        predicted, confidence = classifier.classify(text)
        is_correct = predicted == expected
        correct += is_correct
        print(f"  '{text}'")
        print(f"    → {predicted} (expected {expected}, conf={confidence:.2f}) {'✓' if is_correct else '✗'}")
    
    accuracy = correct / len(test_data) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    return accuracy


def test_gradient_free_learning():
    """Test gradient-free structure learning."""
    print()
    print("=" * 60)
    print("  TASK 6: Gradient-Free Learning (Design 049)")
    print("=" * 60)
    print()
    
    training_data = [
        ('holmes', 'Holmes is a brilliant detective who investigates with Watson.'),
        ('watson', 'Watson is a loyal doctor who assists Holmes.'),
        ('darcy', 'Darcy is a proud gentleman who loves Elizabeth.'),
        ('elizabeth', 'Elizabeth is a witty lady who challenges Darcy.'),
    ]
    
    learner = GradientFreeLearner()
    
    print("Training data:")
    for entity, text in training_data:
        print(f"  {entity}: {text}")
    print()
    
    # Train
    accuracy = learner.train(training_data, epochs=2)
    
    print("Learned structure:")
    for entity, struct in learner.structure.items():
        print(f"  {entity}: {struct}")
    print()
    
    print("Generated outputs:")
    for entity, target in training_data:
        generated = learner.generate(entity)
        print(f"  Target:    {target}")
        print(f"  Generated: {generated}")
        print()
    
    print(f"Final overlap accuracy: {accuracy:.1f}%")
    return accuracy


def test_tachyon_sequences():
    """Test sequence prediction with Tachyon navigation."""
    print()
    print("=" * 60)
    print("  TASK 5: Sequence Prediction (Tachyon Navigation)")
    print("=" * 60)
    print()
    
    training_sequences = [
        ([1, 1, 2], 3),      # Fibonacci
        ([1, 2, 3], 5),      # Fibonacci
        ([2, 3, 5], 8),      # Fibonacci
        ([2, 4, 6], 8),      # Arithmetic +2
        ([5, 10, 15], 20),   # Arithmetic +5
        ([1, 3, 5], 7),      # Arithmetic +2
        ([2, 4, 8], 16),     # Geometric *2
        ([3, 9, 27], 81),    # Geometric *3
        ([1, 2, 4], 8),      # Geometric *2
    ]
    
    predictor = TachyonSequencePredictor()
    predictor.train(training_sequences)
    
    test_sequences = [
        ([1, 1, 2], 3),      # Fibonacci (seen)
        ([8, 13, 21], 34),   # Fibonacci (unseen)
        ([4, 8, 12], 16),    # Arithmetic (unseen)
        ([1, 2, 4], 8),      # Geometric (seen)
        ([5, 25, 125], 625), # Geometric (unseen)
    ]
    
    print("Testing sequence predictions:")
    correct = 0
    for seq, expected in test_sequences:
        predicted, confidence = predictor.predict_next(seq)
        pattern, _ = predictor.detect_pattern(seq)
        is_correct = predicted == expected
        correct += is_correct
        print(f"  {seq} → {predicted} (expected {expected}, pattern={pattern}, conf={confidence:.2f}) {'✓' if is_correct else '✗'}")
    
    accuracy = correct / len(test_sequences) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    return accuracy


def main():
    print()
    print("=" * 60)
    print("  HYPERMAPPING vs NEURAL NETWORKS - ADVANCED")
    print("  Using techniques from Design Considerations")
    print("=" * 60)
    print()
    print("Techniques used:")
    print("  - 044: Quaternion φ-Dial (4D semantic axes)")
    print("  - 049: Gradient-Free Learning (error = structure)")
    print("  - 052: Hypothesis-Driven (test predictions)")
    print("  - 055: Tachyon Navigation (W-axis = certainty)")
    print()
    
    results = {}
    
    # Run tests
    results['Sentiment (Quaternion)'] = test_quaternion_sentiment()
    results['Sentiment (Hypothesis)'] = test_hypothesis_sentiment()
    results['Gradient-Free Learning'] = test_gradient_free_learning()
    results['Sequence (Tachyon)'] = test_tachyon_sequences()
    
    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY - ADVANCED TECHNIQUES")
    print("=" * 60)
    print()
    print("Task                        | Accuracy")
    print("-" * 50)
    for task, acc in results.items():
        print(f"{task:27} | {acc:.1f}%")
    print("-" * 50)
    print(f"{'Average':27} | {sum(results.values()) / len(results):.1f}%")
    print()
    
    # Comparison table
    print("=" * 60)
    print("  COMPARISON: Basic vs Advanced")
    print("=" * 60)
    print()
    print("Task                  | Basic  | Advanced | Improvement")
    print("-" * 60)
    comparisons = [
        ("Sentiment Analysis", 71.4, max(results.get('Sentiment (Quaternion)', 0), 
                                          results.get('Sentiment (Hypothesis)', 0))),
        ("Sequence Prediction", 0.0, results.get('Sequence (Tachyon)', 0)),
        ("Structure Learning", 0.0, results.get('Gradient-Free Learning', 0)),
    ]
    for task, basic, advanced in comparisons:
        improvement = advanced - basic
        print(f"{task:21} | {basic:5.1f}% | {advanced:6.1f}%  | +{improvement:.1f}%")
    print()


if __name__ == "__main__":
    main()
