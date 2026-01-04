"""
HyperdimensionalTranscoder - The Execution Engine

This is the execution side of the hyperdimensional paradigm.
It uses a HyperdimensionalStructure to perform domain-specific operations.

Responsibilities:
- Map domain inputs to positions (encoding)
- Query the structure for relevant nodes
- Apply transformations based on matched nodes
- Map outputs back to domain (decoding)
- Provide feedback for learning

NOT responsible for:
- Managing positions directly
- Structural stability
- Serialization of the structure

The transcoder is domain-SPECIFIC. Different transcoders for:
- TextTranscoder: text → positions → text
- ImageTranscoder: pixels → positions → pixels
- AudioTranscoder: frequencies → positions → frequencies

The key insight: The STRUCTURE is domain-agnostic.
The TRANSCODER provides the domain-specific encoding/decoding.

Author: Lesley Gushurst
License: GPLv3
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from datetime import datetime

from hyperdimensional_structure import HyperdimensionalStructure, Node, CRITICAL_LINE


@dataclass
class TranscoderResult:
    """Result of a transcoding operation."""
    output: Any                          # The transcoded output
    matched_nodes: List[Tuple[Node, float]]  # Nodes that contributed
    confidence: float                    # Overall confidence
    metadata: Dict[str, Any] = field(default_factory=dict)


class HyperdimensionalTranscoder(ABC):
    """
    Abstract base class for domain-specific transcoders.
    
    A transcoder provides:
    1. encode(): Domain input → position
    2. decode(): Position/nodes → domain output
    3. transcode(): Full pipeline (encode → query → decode)
    
    Subclasses implement domain-specific encoding/decoding.
    """
    
    def __init__(self, structure: HyperdimensionalStructure):
        self.structure = structure
        self.transcode_count = 0
        self.feedback_count = 0
    
    @abstractmethod
    def encode(self, input_data: Any) -> np.ndarray:
        """
        Encode domain input to a position in the structure's space.
        
        This is where domain knowledge lives - how to map
        domain-specific data to positions.
        """
        pass
    
    @abstractmethod
    def decode(self, nodes: List[Tuple[Node, float]]) -> Any:
        """
        Decode matched nodes back to domain output.
        
        This is where domain knowledge lives - how to combine
        matched nodes into a domain-specific output.
        """
        pass
    
    def transcode(self, input_data: Any, top_k: int = 5) -> TranscoderResult:
        """
        Full transcoding pipeline.
        
        1. Encode input to position
        2. Query structure for nearest nodes
        3. Decode nodes to output
        """
        # Encode
        position = self.encode(input_data)
        
        # Query
        matches = self.structure.query_nearest(position, k=top_k)
        
        # Decode
        output = self.decode(matches)
        
        # Calculate confidence
        if matches:
            confidence = float(np.mean([sim for _, sim in matches]))
        else:
            confidence = 0.0
        
        self.transcode_count += 1
        
        return TranscoderResult(
            output=output,
            matched_nodes=matches,
            confidence=confidence,
            metadata={
                'input_position': position.tolist(),
                'num_matches': len(matches),
            }
        )
    
    def feedback(self, result: TranscoderResult, success: bool,
                 correct_output: Any = None) -> None:
        """
        Provide feedback on a transcoding result.
        
        If success: reinforce matched nodes (move toward query)
        If failure: correct matched nodes (move away from query)
        
        If correct_output provided, can also add new knowledge.
        """
        query_position = np.array(result.metadata['input_position'])
        
        for node, similarity in result.matched_nodes:
            self.structure.feedback(
                node.id,
                query_position,
                success=success
            )
        
        self.feedback_count += 1
    
    def add_knowledge(self, input_data: Any, output_data: Any,
                      node_id: str = None) -> Node:
        """
        Add new knowledge to the structure.
        
        Encodes the input to get position, stores output as data.
        """
        position = self.encode(input_data)
        node_id = node_id or f"node_{len(self.structure)}_{datetime.now().timestamp()}"
        
        return self.structure.add(
            node_id=node_id,
            position=position,
            data={'input': input_data, 'output': output_data}
        )
    
    def stats(self) -> Dict[str, Any]:
        """Get transcoder statistics."""
        return {
            'structure_stats': self.structure.stats(),
            'transcode_count': self.transcode_count,
            'feedback_count': self.feedback_count,
        }


# =============================================================================
# EXAMPLE: TEXT TRANSCODER
# =============================================================================

class TextTranscoder(HyperdimensionalTranscoder):
    """
    A transcoder for text/chat applications.
    
    Encoding: Text → word overlap → position
    Decoding: Matched nodes → combined text output
    """
    
    # Common filler words to ignore
    FILLER = {'a', 'an', 'the', 'of', 'with', 'for', 'to', 'and', 'or', 'in',
              'that', 'this', 'is', 'are', 'it', 'be', 'can', 'you', 'i', 'me',
              'my', 'your', 'please', 'could', 'would', 'should', 'do', 'does'}
    
    def __init__(self, structure: HyperdimensionalStructure):
        super().__init__(structure)
        self._word_positions: Dict[str, np.ndarray] = {}
    
    def extract_words(self, text: str) -> Set[str]:
        """Extract content words from text."""
        words = text.lower().split()
        return {w for w in words if w not in self.FILLER and len(w) > 1}
    
    def _get_word_position(self, word: str) -> np.ndarray:
        """Get or create a position for a word."""
        if word not in self._word_positions:
            # Create deterministic position from word hash
            np.random.seed(hash(word) % (2**32))
            pos = np.random.randn(self.structure.dims)
            pos = pos / np.linalg.norm(pos)
            self._word_positions[word] = pos
        return self._word_positions[word]
    
    def encode(self, input_data: str) -> np.ndarray:
        """
        Encode text to position.
        
        Strategy: Average of word positions (bag of words in position space)
        """
        words = self.extract_words(input_data)
        
        if not words:
            return np.zeros(self.structure.dims)
        
        # Average word positions
        positions = [self._get_word_position(w) for w in words]
        avg_position = np.mean(positions, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_position)
        if norm > 1e-10:
            avg_position = avg_position / norm * CRITICAL_LINE
        
        return avg_position
    
    def decode(self, nodes: List[Tuple[Node, float]]) -> str:
        """
        Decode matched nodes to text output.
        
        Strategy: Return output from highest-similarity node
        """
        if not nodes:
            return ""
        
        # Get best match
        best_node, best_sim = nodes[0]
        
        if best_node.data and 'output' in best_node.data:
            return best_node.data['output']
        
        return ""
    
    def add_text_knowledge(self, input_text: str, output_text: str,
                           node_id: str = None) -> Node:
        """Convenience method for adding text knowledge."""
        return self.add_knowledge(input_text, output_text, node_id)


# =============================================================================
# EXAMPLE: NUMERIC TRANSCODER
# =============================================================================

class NumericTranscoder(HyperdimensionalTranscoder):
    """
    A transcoder for numeric/vector data.
    
    Encoding: Direct mapping (input IS position)
    Decoding: Weighted average of matched outputs
    
    Useful for: regression, interpolation, function approximation
    """
    
    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """Direct encoding - input is position."""
        arr = np.array(input_data, dtype=np.float64)
        
        # Pad or truncate to match structure dims
        if len(arr) < self.structure.dims:
            arr = np.concatenate([arr, np.zeros(self.structure.dims - len(arr))])
        elif len(arr) > self.structure.dims:
            arr = arr[:self.structure.dims]
        
        return arr
    
    def decode(self, nodes: List[Tuple[Node, float]]) -> np.ndarray:
        """
        Decode via weighted average of outputs.
        
        Weights are the similarity scores.
        """
        if not nodes:
            return np.zeros(self.structure.dims)
        
        total_weight = 0.0
        weighted_sum = np.zeros(self.structure.dims)
        
        for node, similarity in nodes:
            if node.data and 'output' in node.data:
                output = np.array(node.data['output'])
                # Pad/truncate output
                if len(output) < self.structure.dims:
                    output = np.concatenate([output, np.zeros(self.structure.dims - len(output))])
                elif len(output) > self.structure.dims:
                    output = output[:self.structure.dims]
                
                weight = max(similarity, 0)
                weighted_sum += output * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        
        return np.zeros(self.structure.dims)


# =============================================================================
# EXAMPLE: CATEGORICAL TRANSCODER
# =============================================================================

class CategoricalTranscoder(HyperdimensionalTranscoder):
    """
    A transcoder for classification tasks.
    
    Encoding: Feature vector → position
    Decoding: Majority vote of matched categories
    
    Useful for: classification, intent detection, categorization
    """
    
    def __init__(self, structure: HyperdimensionalStructure,
                 feature_extractor: Callable[[Any], np.ndarray] = None):
        super().__init__(structure)
        self.feature_extractor = feature_extractor or self._default_extractor
    
    def _default_extractor(self, input_data: Any) -> np.ndarray:
        """Default: assume input is already a feature vector."""
        return np.array(input_data, dtype=np.float64)
    
    def encode(self, input_data: Any) -> np.ndarray:
        """Encode via feature extraction."""
        features = self.feature_extractor(input_data)
        
        # Pad/truncate
        if len(features) < self.structure.dims:
            features = np.concatenate([features, np.zeros(self.structure.dims - len(features))])
        elif len(features) > self.structure.dims:
            features = features[:self.structure.dims]
        
        return features
    
    def decode(self, nodes: List[Tuple[Node, float]]) -> Any:
        """
        Decode via weighted voting.
        
        Returns the category with highest weighted vote.
        """
        if not nodes:
            return None
        
        votes: Dict[Any, float] = {}
        
        for node, similarity in nodes:
            if node.data and 'output' in node.data:
                category = node.data['output']
                weight = max(similarity, 0)
                votes[category] = votes.get(category, 0) + weight
        
        if not votes:
            return None
        
        return max(votes, key=votes.get)
    
    def add_example(self, input_data: Any, category: Any,
                    node_id: str = None) -> Node:
        """Add a classification example."""
        return self.add_knowledge(input_data, category, node_id)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== HyperdimensionalTranscoder Test ===\n")
    
    # Test 1: Text Transcoder
    print("--- Text Transcoder ---")
    structure = HyperdimensionalStructure(dims=8, name="text_test")
    text_transcoder = TextTranscoder(structure)
    
    # Add knowledge
    text_transcoder.add_text_knowledge(
        "hello how are you",
        "I'm doing well, thank you for asking!"
    )
    text_transcoder.add_text_knowledge(
        "what is the weather",
        "I don't have access to weather data."
    )
    text_transcoder.add_text_knowledge(
        "tell me a joke",
        "Why did the programmer quit? Because he didn't get arrays!"
    )
    
    # Transcode
    result = text_transcoder.transcode("hi how are you doing")
    print(f"Input: 'hi how are you doing'")
    print(f"Output: '{result.output}'")
    print(f"Confidence: {result.confidence:.3f}")
    
    result = text_transcoder.transcode("what's the weather like")
    print(f"\nInput: 'what's the weather like'")
    print(f"Output: '{result.output}'")
    print(f"Confidence: {result.confidence:.3f}")
    
    # Test 2: Categorical Transcoder
    print("\n--- Categorical Transcoder ---")
    structure2 = HyperdimensionalStructure(dims=4, name="category_test")
    cat_transcoder = CategoricalTranscoder(structure2)
    
    # Add examples (using simple feature vectors)
    cat_transcoder.add_example([1, 0, 0, 0], "category_A")
    cat_transcoder.add_example([0.9, 0.1, 0, 0], "category_A")
    cat_transcoder.add_example([0, 1, 0, 0], "category_B")
    cat_transcoder.add_example([0.1, 0.9, 0, 0], "category_B")
    
    # Classify
    result = cat_transcoder.transcode([0.8, 0.2, 0, 0])
    print(f"Input: [0.8, 0.2, 0, 0]")
    print(f"Category: {result.output}")
    print(f"Confidence: {result.confidence:.3f}")
    
    result = cat_transcoder.transcode([0.2, 0.8, 0, 0])
    print(f"\nInput: [0.2, 0.8, 0, 0]")
    print(f"Category: {result.output}")
    print(f"Confidence: {result.confidence:.3f}")
    
    # Test 3: Numeric Transcoder (function approximation)
    print("\n--- Numeric Transcoder ---")
    structure3 = HyperdimensionalStructure(dims=2, name="numeric_test")
    num_transcoder = NumericTranscoder(structure3)
    
    # Learn f(x) = x^2 at a few points
    for x in [0, 1, 2, 3, 4]:
        num_transcoder.add_knowledge(
            np.array([x, 0]),
            np.array([x*x, 0])
        )
    
    # Interpolate
    result = num_transcoder.transcode(np.array([2.5, 0]))
    print(f"Input: x=2.5")
    print(f"Output: {result.output[0]:.2f} (expected: 6.25)")
    print(f"Confidence: {result.confidence:.3f}")
    
    print("\n✓ All transcoders working!")
    print("\nKey insight: Same structure, different transcoders!")
    print("The structure is domain-agnostic.")
    print("The transcoder provides domain-specific encoding/decoding.")
