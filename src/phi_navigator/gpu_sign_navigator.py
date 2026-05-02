"""
GPU Sign Navigator - Geometric navigation using sign matrices on GPU.

Key insight from Doc 141 (Irreducible Shape):
- Signs are NOT learned statistics - they're the GEOMETRIC STRUCTURE
- 3584 critical lines (hyperplanes) divide semantic space
- Each sign bit = which side of a critical line
- This IS the irreducible shape of knowledge

Key insight from Doc 095 (HyperMapping):
- Neural networks ARE geometry
- Attention IS nearest-neighbor search
- FFN IS position transformation
- We just make it explicit

GPU Strategy:
- Pack signs as int8 (+1/-1) or even bits
- Sign multiplication = XOR-like operation
- Batched matmul for multiple layers at once
- Eliminate memory bandwidth by fusing operations

The forward pass becomes:
1. Encode: input → sign pattern (which region of space?)
2. Transform: sign_out = sign(W_signs @ input_signs) 
3. Navigate: find nearest token by Hamming distance

All operations are integer/bitwise - perfect for GPU.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PHI = 1.6180339887498949


class SignTensor:
    """
    Sign tensor for GPU computation.
    
    Signs are stored as int8 (+1/-1) for easy multiplication.
    Can be packed to bits for storage (32x compression).
    
    The key insight: sign multiplication IS the geometric operation.
    W @ x in sign space = which critical lines does x cross?
    """
    
    def __init__(self, signs: np.ndarray, device: str = 'cpu'):
        """
        Args:
            signs: int8 array with values in {-1, +1}
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        if HAS_TORCH and device != 'cpu':
            self.data = torch.tensor(signs, dtype=torch.int8, device=device)
            self._is_torch = True
        else:
            self.data = signs.astype(np.int8)
            self._is_torch = False
    
    @classmethod
    def from_weights(cls, W: np.ndarray, device: str = 'cpu') -> 'SignTensor':
        """Extract signs from weight matrix."""
        signs = np.sign(W).astype(np.int8)
        signs[signs == 0] = 1
        return cls(signs, device)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)
    
    def to_numpy(self) -> np.ndarray:
        if self._is_torch:
            return self.data.cpu().numpy()
        return self.data
    
    def matmul_sign(self, x: 'SignTensor') -> 'SignTensor':
        """
        Sign-space matrix multiplication.
        
        For each output dimension i:
            out[i] = sign(sum_j(W[i,j] * x[j]))
        
        This computes: which side of each critical line does the result fall?
        
        The sum of products of signs gives a "vote" - majority wins.
        """
        if self._is_torch:
            # Use float for matmul then convert back
            W = self.data.float()
            x_data = x.data.float()
            result = torch.sign(W @ x_data).to(torch.int8)
            result[result == 0] = 1
            return SignTensor(result.cpu().numpy(), self.device)
        else:
            # NumPy path
            W = self.data.astype(np.float32)
            x_data = x.data.astype(np.float32)
            result = np.sign(W @ x_data).astype(np.int8)
            result[result == 0] = 1
            return SignTensor(result, self.device)
    
    def hadamard(self, other: 'SignTensor') -> 'SignTensor':
        """Element-wise sign multiplication (like XOR for ±1)."""
        if self._is_torch:
            result = (self.data * other.data).to(torch.int8)
            return SignTensor(result.cpu().numpy(), self.device)
        else:
            result = (self.data * other.data).astype(np.int8)
            return SignTensor(result, self.device)
    
    def hamming_distance(self, other: 'SignTensor') -> int:
        """Count disagreements (Hamming distance in sign space)."""
        if self._is_torch:
            return int((self.data != other.data).sum().item())
        else:
            return int(np.sum(self.data != other.data))
    
    def agreement_score(self, other: 'SignTensor') -> float:
        """Fraction of signs that agree (1.0 = identical)."""
        if self._is_torch:
            return float((self.data == other.data).float().mean().item())
        else:
            return float(np.mean(self.data == other.data))


class BatchedSignTransform:
    """
    Batched sign transformations for GPU efficiency.
    
    Instead of:
        for layer in layers:
            x = layer(x)  # Memory transfer each time
    
    We do:
        x = batched_transform(x, all_layers)  # One kernel launch
    
    This eliminates the memory bandwidth bottleneck.
    """
    
    def __init__(self, transforms: List[SignTensor], device: str = 'cpu'):
        """
        Args:
            transforms: List of sign matrices (one per layer)
            device: 'cpu' or 'cuda'
        """
        self.transforms = transforms
        self.device = device
        self.n_layers = len(transforms)
        
        # For truly batched execution, we'd fuse these into a single tensor
        # But that requires all layers to have compatible shapes
        # For now, we batch what we can
    
    def forward_all(self, x: SignTensor) -> List[SignTensor]:
        """
        Apply all transforms and return intermediate results.
        
        This is useful for debugging and understanding the geometry.
        """
        results = [x]
        current = x
        for transform in self.transforms:
            current = transform.matmul_sign(current)
            results.append(current)
        return results
    
    def forward(self, x: SignTensor) -> SignTensor:
        """Apply all transforms, return final result."""
        current = x
        for transform in self.transforms:
            current = transform.matmul_sign(current)
        return current


class GPUSignNavigator:
    """
    GPU-accelerated sign-based geometric navigator.
    
    Architecture:
    1. Embeddings as sign patterns (which region of semantic space?)
    2. Layers as sign transforms (which critical lines to cross?)
    3. Output as nearest neighbor in sign space (Hamming distance)
    
    The forward pass is entirely geometric:
    - No floating point (except for intermediate sums)
    - No learned parameters (signs ARE the geometry)
    - O(1) memory per layer (signs are 1 bit)
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Embeddings: (vocab_size, hidden_dim) sign patterns
        self.embedding_signs: Optional[SignTensor] = None
        
        # Layer transforms
        self.layer_transforms: List[Dict[str, SignTensor]] = []
        
        # LM head signs
        self.lm_head_signs: Optional[SignTensor] = None
        
        # Tokenizer
        self.tokenizer = None
        
        # Config
        self.hidden_dim = None
        self.vocab_size = None
        self.intermediate_dim = None
    
    def convert_from_model(self, model_name: str = "Qwen/Qwen2-7B-Instruct",
                           max_layers: int = 28):
        """Convert model to sign-based representation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        self.intermediate_dim = model.config.intermediate_size
        
        # Convert embeddings
        print("Converting embeddings...")
        emb_weight = model.model.embed_tokens.weight.data.numpy()
        self.embedding_signs = SignTensor.from_weights(emb_weight, self.device)
        
        # Convert LM head
        print("Converting LM head...")
        lm_weight = model.lm_head.weight.data.numpy()
        self.lm_head_signs = SignTensor.from_weights(lm_weight, self.device)
        
        # Convert layers
        n_layers = min(max_layers, len(model.model.layers))
        print(f"Converting {n_layers} layers...")
        
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            layer_dict = {}
            
            # MLP signs (the main transformation)
            layer_dict['gate'] = SignTensor.from_weights(
                layer.mlp.gate_proj.weight.data.numpy(), self.device
            )
            layer_dict['up'] = SignTensor.from_weights(
                layer.mlp.up_proj.weight.data.numpy(), self.device
            )
            layer_dict['down'] = SignTensor.from_weights(
                layer.mlp.down_proj.weight.data.numpy(), self.device
            )
            
            # Attention signs (for relationship encoding)
            layer_dict['q'] = SignTensor.from_weights(
                layer.self_attn.q_proj.weight.data.numpy(), self.device
            )
            layer_dict['k'] = SignTensor.from_weights(
                layer.self_attn.k_proj.weight.data.numpy(), self.device
            )
            layer_dict['v'] = SignTensor.from_weights(
                layer.self_attn.v_proj.weight.data.numpy(), self.device
            )
            layer_dict['o'] = SignTensor.from_weights(
                layer.self_attn.o_proj.weight.data.numpy(), self.device
            )
            
            self.layer_transforms.append(layer_dict)
            
            if layer_idx % 5 == 0:
                print(f"  Layer {layer_idx} converted")
        
        print(f"Converted {n_layers} layers")
        
        # Calculate compression
        original_bytes = (
            emb_weight.nbytes + lm_weight.nbytes +
            sum(
                layer.mlp.gate_proj.weight.data.numpy().nbytes +
                layer.mlp.up_proj.weight.data.numpy().nbytes +
                layer.mlp.down_proj.weight.data.numpy().nbytes +
                layer.self_attn.q_proj.weight.data.numpy().nbytes +
                layer.self_attn.k_proj.weight.data.numpy().nbytes +
                layer.self_attn.v_proj.weight.data.numpy().nbytes +
                layer.self_attn.o_proj.weight.data.numpy().nbytes
                for layer in model.model.layers[:n_layers]
            )
        )
        # Signs are 1 byte each (could be 1 bit with packing)
        sign_bytes = (
            self.embedding_signs.data.nbytes +
            self.lm_head_signs.data.nbytes +
            sum(
                layer_dict['gate'].data.nbytes +
                layer_dict['up'].data.nbytes +
                layer_dict['down'].data.nbytes +
                layer_dict['q'].data.nbytes +
                layer_dict['k'].data.nbytes +
                layer_dict['v'].data.nbytes +
                layer_dict['o'].data.nbytes
                for layer_dict in self.layer_transforms
            )
        )
        
        print(f"\nCompression: {original_bytes / sign_bytes:.1f}x (int8)")
        print(f"  With bit packing: {original_bytes / (sign_bytes / 8):.1f}x")
        
        del model
    
    def get_embedding_signs(self, token_id: int) -> SignTensor:
        """Get sign pattern for a token."""
        signs = self.embedding_signs.to_numpy()[token_id]
        return SignTensor(signs, self.device)
    
    def find_nearest_token(self, target: SignTensor) -> Tuple[int, float]:
        """
        Find token with most similar sign pattern.
        
        This is the geometric query: which point in embedding space
        is closest to our current position (in Hamming distance)?
        """
        target_np = target.to_numpy()
        emb_np = self.embedding_signs.to_numpy()
        
        # Compute agreement with all tokens
        # agreement[i] = fraction of signs that match
        agreement = np.mean(emb_np == target_np, axis=1)
        
        best_idx = int(np.argmax(agreement))
        best_score = float(agreement[best_idx])
        
        return best_idx, best_score
    
    def forward_mlp_signs(self, x: SignTensor, layer_idx: int) -> SignTensor:
        """
        MLP forward pass in sign space.
        
        Standard MLP: out = down(silu(gate(x)) * up(x))
        
        Sign-space MLP:
        1. gate_signs = sign(gate @ x)
        2. up_signs = sign(up @ x)
        3. hidden_signs = gate_signs * up_signs  (element-wise)
        4. out_signs = sign(down @ hidden_signs)
        
        This preserves the STRUCTURE of the MLP while operating on signs.
        """
        layer = self.layer_transforms[layer_idx]
        
        # Gate and up projections
        gate_out = layer['gate'].matmul_sign(x)
        up_out = layer['up'].matmul_sign(x)
        
        # Element-wise multiplication (the "gating")
        hidden = gate_out.hadamard(up_out)
        
        # Down projection
        out = layer['down'].matmul_sign(hidden)
        
        return out
    
    def forward_attention_signs(self, x: SignTensor, layer_idx: int) -> SignTensor:
        """
        Attention forward pass in sign space.
        
        For single-token (no KV cache), attention simplifies significantly.
        
        Qwen2-7B uses GQA (Grouped Query Attention):
        - Q: (3584, 3584) - 28 heads × 128 head_dim
        - K: (512, 3584) - 4 KV heads × 128 head_dim  
        - V: (512, 3584) - 4 KV heads × 128 head_dim
        - O: (3584, 3584) - projects back to hidden_dim
        
        For single token, we skip Q/K (no cross-attention) and just use Q→O path.
        This is a simplification but preserves the geometric structure.
        """
        layer = self.layer_transforms[layer_idx]
        
        # For single token, use Q projection directly to O
        # Q: (3584, 3584) @ (3584,) -> (3584,)
        # O: (3584, 3584) @ (3584,) -> (3584,)
        q_out = layer['q'].matmul_sign(x)
        out = layer['o'].matmul_sign(q_out)
        
        return out
    
    def forward_layer_signs(self, x: SignTensor, layer_idx: int) -> SignTensor:
        """
        Full layer forward pass in sign space.
        
        Transformer layer: out = x + attn(norm(x)) + mlp(norm(x + attn))
        
        Sign-space layer (simplified):
        1. attn_out = attention_signs(x)
        2. x = x XOR attn_out (residual in sign space)
        3. mlp_out = mlp_signs(x)
        4. x = x XOR mlp_out
        
        The XOR-like residual: if both agree, keep sign; if disagree, flip.
        Actually: sign(x + residual) ≈ sign with larger magnitude wins
        
        For pure sign space, we use majority vote:
        out[i] = sign(x[i] + attn[i] + mlp[i])
        """
        # Attention
        attn_out = self.forward_attention_signs(x, layer_idx)
        
        # MLP
        mlp_out = self.forward_mlp_signs(x, layer_idx)
        
        # Residual combination: majority vote of (x, attn, mlp)
        # If 2+ agree, that's the output sign
        x_np = x.to_numpy().astype(np.int16)
        attn_np = attn_out.to_numpy().astype(np.int16)
        mlp_np = mlp_out.to_numpy().astype(np.int16)
        
        combined = x_np + attn_np + mlp_np
        out_signs = np.sign(combined).astype(np.int8)
        out_signs[out_signs == 0] = 1
        
        return SignTensor(out_signs, self.device)
    
    def navigate(self, token_ids: List[int]) -> SignTensor:
        """Navigate through model using sign-only computation."""
        # Start with last token's signs
        position = self.get_embedding_signs(token_ids[-1])
        
        # Navigate through layers
        for layer_idx in range(len(self.layer_transforms)):
            position = self.forward_layer_signs(position, layer_idx)
        
        return position
    
    def navigate_hamming(self, token_ids: List[int]) -> Tuple[int, float]:
        """
        Alternative navigation: find token whose embedding signs,
        when transformed through layers, best match the input.
        
        This is more like HyperMapping's approach:
        - Each token defines a region of sign space
        - Navigation finds which region we end up in
        """
        # Get input token's signs
        input_signs = self.get_embedding_signs(token_ids[-1])
        input_np = input_signs.to_numpy()
        
        # For each candidate token, compute how well it "explains" the input
        # after accounting for the layer transformations
        emb_np = self.embedding_signs.to_numpy()
        
        # Simple approach: find token most similar to input
        # (The layers should transform similar inputs to similar outputs)
        agreement = np.mean(emb_np == input_np, axis=1)
        
        # Exclude the input token itself
        agreement[token_ids[-1]] = 0
        
        best_idx = int(np.argmax(agreement))
        best_score = float(agreement[best_idx])
        
        return best_idx, best_score
    
    def predict_next_token(self, token_ids: List[int], method: str = 'transform') -> Tuple[int, float]:
        """
        Predict next token via geometric navigation.
        
        Methods:
        - 'transform': Apply sign transforms then find nearest
        - 'hamming': Direct Hamming distance navigation
        """
        if method == 'hamming':
            return self.navigate_hamming(token_ids)
        else:
            output_signs = self.navigate(token_ids)
            return self.find_nearest_token(output_signs)
    
    def generate(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate text using sign-only navigation."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        generated = []
        for _ in range(max_tokens):
            next_token, score = self.predict_next_token(token_ids)
            token_ids.append(next_token)
            generated.append((next_token, score))
            
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(token_ids)


def test_sign_tensor():
    """Test SignTensor operations."""
    print("=" * 60)
    print("Testing SignTensor")
    print("=" * 60)
    
    # Create random sign tensors
    np.random.seed(42)
    W = np.random.randn(256, 512).astype(np.float32)
    x = np.random.randn(512).astype(np.float32)
    
    W_signs = SignTensor.from_weights(W)
    x_signs = SignTensor.from_weights(x)
    
    print(f"W shape: {W_signs.shape}")
    print(f"x shape: {x_signs.shape}")
    
    # Test matmul
    result = W_signs.matmul_sign(x_signs)
    print(f"Result shape: {result.shape}")
    print(f"Result unique values: {np.unique(result.to_numpy())}")
    
    # Compare to float matmul
    float_result = np.sign(W @ x)
    float_result[float_result == 0] = 1
    
    agreement = np.mean(result.to_numpy() == float_result.astype(np.int8))
    print(f"Agreement with float matmul: {agreement:.2%}")
    print()


def test_batched_transform():
    """Test batched sign transformations."""
    print("=" * 60)
    print("Testing BatchedSignTransform")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create a sequence of transforms (like layers)
    dims = [512, 256, 256, 512]
    transforms = []
    for i in range(len(dims) - 1):
        W = np.random.randn(dims[i+1], dims[i]).astype(np.float32)
        transforms.append(SignTensor.from_weights(W))
    
    batched = BatchedSignTransform(transforms)
    
    # Input
    x = np.random.randn(dims[0]).astype(np.float32)
    x_signs = SignTensor.from_weights(x)
    
    # Forward
    results = batched.forward_all(x_signs)
    
    print(f"Input shape: {x_signs.shape}")
    print(f"Number of transforms: {batched.n_layers}")
    for i, r in enumerate(results):
        print(f"  After transform {i}: shape {r.shape}")
    
    final = batched.forward(x_signs)
    print(f"Final shape: {final.shape}")
    print()


def test_geometric_navigation():
    """Test the geometric navigation concept."""
    print("=" * 60)
    print("Testing Geometric Navigation Concept")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate a small vocabulary
    vocab_size = 100
    hidden_dim = 64
    
    # Create "embeddings" as sign patterns
    embeddings = np.random.randn(vocab_size, hidden_dim).astype(np.float32)
    emb_signs = SignTensor.from_weights(embeddings)
    
    # Create a "transform" (like a layer)
    W = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
    W_signs = SignTensor.from_weights(W)
    
    # Pick a token and navigate
    token_id = 42
    token_signs = SignTensor(emb_signs.to_numpy()[token_id], 'cpu')
    
    print(f"Starting token: {token_id}")
    print(f"Token signs shape: {token_signs.shape}")
    
    # Apply transform
    transformed = W_signs.matmul_sign(token_signs)
    print(f"Transformed shape: {transformed.shape}")
    
    # Find nearest token
    emb_np = emb_signs.to_numpy()
    trans_np = transformed.to_numpy()
    
    agreement = np.mean(emb_np == trans_np, axis=1)
    nearest = np.argmax(agreement)
    
    print(f"Nearest token after transform: {nearest}")
    print(f"Agreement score: {agreement[nearest]:.2%}")
    print(f"Self-agreement (token 42): {agreement[42]:.2%}")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        navigator = GPUSignNavigator(device='cpu')
        navigator.convert_from_model(max_layers=2)
        
        prompt = "Hello"
        print(f"\nPrompt: {prompt}")
        output = navigator.generate(prompt, max_tokens=10)
        print(f"Output: {output}")
    else:
        test_sign_tensor()
        test_batched_transform()
        test_geometric_navigation()
