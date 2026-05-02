"""
φ-Space Memory: Persistent Memory as Geometric Locations

Core insight: Memory IS position in φ-space.
- Storing a memory = recording a location
- Retrieving a memory = navigating to that region
- Similar memories = nearby locations

Human-readable inspection:
- Nearest concept neighbors
- Dimension activations (if interpretable)
- Token decoding (what would this embedding say?)
- Semantic tags derived from geometry
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib

# Constants
PHI = 1.6180339887498949

@dataclass
class MemoryEntry:
    """A single memory stored in φ-space."""
    id: str
    content: str  # Human-readable content
    embedding: List[float]  # φ-space location
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    
    # Human-readable interpretation (computed on storage)
    nearest_concepts: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)
    phi_signature: float = 0.0  # φ-level of this memory


class PhiMemory:
    """
    Persistent memory system using φ-space embeddings.
    
    Memories are stored as locations in geometric space.
    Retrieval is navigation - finding nearby points.
    """
    
    def __init__(self, model, tokenizer, storage_path: str = "phi_memory.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.storage_path = Path(storage_path)
        self.memories: Dict[str, MemoryEntry] = {}
        
        # Reference concepts for interpretation
        self.reference_concepts = [
            "mathematics", "science", "art", "language", "logic",
            "emotion", "action", "object", "person", "place",
            "time", "cause", "effect", "question", "answer",
            "problem", "solution", "method", "result", "error",
            "number", "equation", "proof", "theorem", "formula",
            "code", "function", "variable", "loop", "condition"
        ]
        self.reference_embeddings: Optional[torch.Tensor] = None
        
        # Load existing memories
        self._load()
        
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get the φ-space embedding for text."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
            # Use layer 27 (the bottleneck) if available, else last layer
            hidden_states = outputs.hidden_states
            layer_idx = min(27, len(hidden_states) - 1)
            embedding = hidden_states[layer_idx][0, -1, :]  # Last token
            
        return embedding
    
    def _compute_reference_embeddings(self):
        """Compute embeddings for reference concepts (for interpretation)."""
        if self.reference_embeddings is not None:
            return
            
        embeddings = []
        for concept in self.reference_concepts:
            emb = self._get_embedding(concept)
            embeddings.append(emb)
        
        self.reference_embeddings = torch.stack(embeddings)
    
    def _interpret_embedding(self, embedding: torch.Tensor) -> Tuple[List[str], List[str], float]:
        """
        Make an embedding human-readable.
        
        Returns:
            - nearest_concepts: Top 5 closest reference concepts
            - semantic_tags: Derived semantic categories
            - phi_signature: The φ-level of this embedding
        """
        self._compute_reference_embeddings()
        
        # Compute similarities to reference concepts
        embedding_norm = embedding / embedding.norm()
        ref_norms = self.reference_embeddings / self.reference_embeddings.norm(dim=1, keepdim=True)
        similarities = torch.matmul(ref_norms, embedding_norm)
        
        # Get top 5 nearest concepts
        top_k = min(5, len(self.reference_concepts))
        top_indices = similarities.topk(top_k).indices.tolist()
        nearest_concepts = [self.reference_concepts[i] for i in top_indices]
        
        # Derive semantic tags based on concept clusters
        semantic_tags = []
        math_concepts = {"mathematics", "number", "equation", "proof", "theorem", "formula"}
        code_concepts = {"code", "function", "variable", "loop", "condition"}
        reasoning_concepts = {"logic", "cause", "effect", "problem", "solution", "method"}
        
        nearest_set = set(nearest_concepts)
        if nearest_set & math_concepts:
            semantic_tags.append("mathematical")
        if nearest_set & code_concepts:
            semantic_tags.append("computational")
        if nearest_set & reasoning_concepts:
            semantic_tags.append("reasoning")
        if "question" in nearest_set or "answer" in nearest_set:
            semantic_tags.append("qa")
        if "error" in nearest_set or "result" in nearest_set:
            semantic_tags.append("outcome")
            
        # Compute φ-signature (ratio of top eigenvalues approximation)
        # Using the embedding's self-similarity structure
        emb_np = embedding.cpu().float().numpy()
        # Simple φ-signature: ratio of L2 norm to L1 norm (bounded measure)
        l1 = np.abs(emb_np).sum()
        l2 = np.sqrt((emb_np ** 2).sum())
        phi_signature = float(l1 / (l2 * np.sqrt(len(emb_np)))) if l2 > 0 else 0.0
        
        return nearest_concepts, semantic_tags, phi_signature
    
    def _decode_embedding(self, embedding: torch.Tensor, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Decode an embedding back to tokens using lm_head.
        
        This shows "what would this embedding say?" - making the
        geometric location human-readable.
        """
        with torch.no_grad():
            # Project through lm_head to get token logits
            # Ensure embedding matches model dtype
            embedding = embedding.to(self.model.lm_head.weight.dtype)
            logits = self.model.lm_head(embedding.unsqueeze(0).unsqueeze(0))
            probs = torch.softmax(logits[0, 0], dim=-1)
            
            # Get top tokens
            top_probs, top_indices = probs.topk(top_k)
            
            tokens = []
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                token = self.tokenizer.decode([idx])
                tokens.append((token, prob))
                
        return tokens
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a memory."""
        hash_input = f"{content}{datetime.now().isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    
    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """
        Store a memory in φ-space.
        
        The memory is embedded and its location is recorded along with
        human-readable interpretations.
        """
        # Get embedding
        embedding = self._get_embedding(content)
        
        # Interpret the embedding
        nearest_concepts, semantic_tags, phi_signature = self._interpret_embedding(embedding)
        
        # Create memory entry
        memory = MemoryEntry(
            id=self._generate_id(content),
            content=content,
            embedding=embedding.cpu().tolist(),
            metadata=metadata or {},
            nearest_concepts=nearest_concepts,
            semantic_tags=semantic_tags,
            phi_signature=phi_signature
        )
        
        self.memories[memory.id] = memory
        self._save()
        
        return memory
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve memories by navigating to the query's location in φ-space.
        
        Returns memories sorted by geometric proximity (cosine similarity).
        """
        if not self.memories:
            return []
            
        # Get query embedding
        query_embedding = self._get_embedding(query)
        query_norm = query_embedding / query_embedding.norm()
        
        # Compute similarities to all memories
        results = []
        for memory in self.memories.values():
            mem_embedding = torch.tensor(memory.embedding, device=query_embedding.device, dtype=query_embedding.dtype)
            mem_norm = mem_embedding / mem_embedding.norm()
            similarity = float(torch.dot(query_norm, mem_norm))
            results.append((memory, similarity))
            
            # Update access count
            memory.access_count += 1
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        self._save()
        return results[:top_k]
    
    def inspect(self, memory_id: str) -> Dict[str, Any]:
        """
        Get a detailed human-readable inspection of a memory.
        
        This is the key function for examining embedding contents!
        """
        if memory_id not in self.memories:
            return {"error": f"Memory {memory_id} not found"}
            
        memory = self.memories[memory_id]
        embedding = torch.tensor(memory.embedding, device=next(self.model.parameters()).device)
        
        # Decode to tokens
        decoded_tokens = self._decode_embedding(embedding)
        
        return {
            "id": memory.id,
            "content": memory.content,
            "created_at": memory.created_at,
            "access_count": memory.access_count,
            
            # Human-readable interpretation
            "interpretation": {
                "nearest_concepts": memory.nearest_concepts,
                "semantic_tags": memory.semantic_tags,
                "phi_signature": memory.phi_signature,
                
                # What tokens would this embedding produce?
                "decoded_tokens": [
                    {"token": tok, "probability": f"{prob:.4f}"}
                    for tok, prob in decoded_tokens
                ],
                
                # Embedding statistics
                "embedding_stats": {
                    "dimensions": len(memory.embedding),
                    "mean": float(np.mean(memory.embedding)),
                    "std": float(np.std(memory.embedding)),
                    "min": float(np.min(memory.embedding)),
                    "max": float(np.max(memory.embedding)),
                }
            },
            
            "metadata": memory.metadata
        }
    
    def inspect_all(self) -> List[Dict[str, Any]]:
        """Get a summary of all memories."""
        summaries = []
        for memory in self.memories.values():
            summaries.append({
                "id": memory.id,
                "content": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                "nearest_concepts": memory.nearest_concepts[:3],
                "semantic_tags": memory.semantic_tags,
                "phi_signature": f"{memory.phi_signature:.4f}",
                "access_count": memory.access_count
            })
        return summaries
    
    def compare(self, memory_id1: str, memory_id2: str) -> Dict[str, Any]:
        """Compare two memories geometrically."""
        if memory_id1 not in self.memories or memory_id2 not in self.memories:
            return {"error": "Memory not found"}
            
        m1 = self.memories[memory_id1]
        m2 = self.memories[memory_id2]
        
        e1 = torch.tensor(m1.embedding)
        e2 = torch.tensor(m2.embedding)
        
        # Cosine similarity
        similarity = float(torch.dot(e1/e1.norm(), e2/e2.norm()))
        
        # Euclidean distance
        distance = float((e1 - e2).norm())
        
        # Difference vector interpretation
        diff = e2 - e1
        diff_tensor = diff.to(next(self.model.parameters()).device)
        
        return {
            "memory_1": {"id": m1.id, "content": m1.content[:50]},
            "memory_2": {"id": m2.id, "content": m2.content[:50]},
            "similarity": similarity,
            "distance": distance,
            "shared_concepts": list(set(m1.nearest_concepts) & set(m2.nearest_concepts)),
            "unique_to_1": list(set(m1.nearest_concepts) - set(m2.nearest_concepts)),
            "unique_to_2": list(set(m2.nearest_concepts) - set(m1.nearest_concepts)),
        }
    
    def _save(self):
        """Save memories to disk."""
        data = {
            mid: asdict(mem) for mid, mem in self.memories.items()
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load memories from disk."""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            for mid, mem_dict in data.items():
                self.memories[mid] = MemoryEntry(**mem_dict)
            print(f"Loaded {len(self.memories)} memories from {self.storage_path}")


def demo():
    """Demonstrate φ-space memory with human-readable inspection."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Create memory system
    memory = PhiMemory(model, tokenizer, storage_path="demo_phi_memory.json")
    
    print("\n" + "="*60)
    print("φ-SPACE MEMORY DEMO")
    print("="*60)
    
    # Store some memories
    print("\n--- Storing Memories ---")
    
    memories_to_store = [
        "The sum of integers from 1 to 100 is 5050, computed using Gauss's formula n(n+1)/2",
        "To solve a quadratic equation ax²+bx+c=0, use the quadratic formula: x = (-b ± √(b²-4ac)) / 2a",
        "Python list comprehensions are a concise way to create lists: [x**2 for x in range(10)]",
        "The derivative of x^n is n*x^(n-1), a fundamental rule of calculus",
        "Debugging tip: always check your assumptions first, then trace the data flow"
    ]
    
    stored = []
    for content in memories_to_store:
        mem = memory.store(content)
        stored.append(mem)
        print(f"\nStored: {content[:50]}...")
        print(f"  ID: {mem.id}")
        print(f"  Nearest concepts: {mem.nearest_concepts}")
        print(f"  Semantic tags: {mem.semantic_tags}")
        print(f"  φ-signature: {mem.phi_signature:.4f}")
    
    # Retrieve by query
    print("\n" + "="*60)
    print("--- Retrieval by Query ---")
    print("="*60)
    
    queries = [
        "How do I add up numbers?",
        "What's the formula for solving x squared equations?",
        "How do I write efficient Python code?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = memory.retrieve(query, top_k=2)
        for mem, sim in results:
            print(f"  [{sim:.4f}] {mem.content[:60]}...")
    
    # Detailed inspection
    print("\n" + "="*60)
    print("--- Detailed Memory Inspection ---")
    print("="*60)
    
    if stored:
        inspection = memory.inspect(stored[0].id)
        print(f"\nInspecting memory: {inspection['id']}")
        print(f"Content: {inspection['content'][:80]}...")
        print(f"\nInterpretation:")
        print(f"  Nearest concepts: {inspection['interpretation']['nearest_concepts']}")
        print(f"  Semantic tags: {inspection['interpretation']['semantic_tags']}")
        print(f"  φ-signature: {inspection['interpretation']['phi_signature']:.4f}")
        print(f"\n  Decoded tokens (what this embedding 'says'):")
        for tok_info in inspection['interpretation']['decoded_tokens'][:5]:
            print(f"    '{tok_info['token']}' ({tok_info['probability']})")
        print(f"\n  Embedding stats:")
        stats = inspection['interpretation']['embedding_stats']
        print(f"    Dimensions: {stats['dimensions']}")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Compare memories
    print("\n" + "="*60)
    print("--- Memory Comparison ---")
    print("="*60)
    
    if len(stored) >= 2:
        comparison = memory.compare(stored[0].id, stored[1].id)
        print(f"\nComparing:")
        print(f"  Memory 1: {comparison['memory_1']['content']}...")
        print(f"  Memory 2: {comparison['memory_2']['content']}...")
        print(f"\n  Similarity: {comparison['similarity']:.4f}")
        print(f"  Distance: {comparison['distance']:.4f}")
        print(f"  Shared concepts: {comparison['shared_concepts']}")
        print(f"  Unique to 1: {comparison['unique_to_1']}")
        print(f"  Unique to 2: {comparison['unique_to_2']}")
    
    # Show all memories
    print("\n" + "="*60)
    print("--- All Memories Summary ---")
    print("="*60)
    
    for summary in memory.inspect_all():
        print(f"\n  [{summary['id']}] φ={summary['phi_signature']}")
        print(f"    {summary['content']}")
        print(f"    Tags: {summary['semantic_tags']}, Concepts: {summary['nearest_concepts']}")


if __name__ == "__main__":
    demo()
