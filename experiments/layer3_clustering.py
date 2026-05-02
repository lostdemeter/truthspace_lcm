#!/usr/bin/env python3
"""
Layer 3 Clustering: Can We Compress the Click?
================================================

Key findings so far:
- Layer 3 is the "click point" (cosine drops 0.62 → 0.10)
- Context always matters (0% single-token accuracy)
- h3 = h2 + attn_output + mlp_output (perfect reconstruction)

Questions:
1. Do layer 3 outputs cluster into types?
2. Can we predict the cluster from (A, B)?
3. If yes, we can cache cluster centroids instead of all pairs

Author: TruthSpace LCM Team
Date: 2026-02-02
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class Layer3ClusteringAnalyzer:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        print(f"Loading model: {model_name}")
        
        config = AutoConfig.from_pretrained(model_name)
        config._attn_implementation = "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        
        self.hidden_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        
        self.lm_head = self.model.lm_head.weight.data.float().cpu().numpy()
        
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.n_layers}")
    
    def collect_layer3_outputs(self, n_samples: int = 500):
        """Collect layer 3 outputs for random (A, B) pairs."""
        print(f"\n--- Collecting Layer 3 Outputs ({n_samples} pairs) ---")
        
        h3_outputs = []
        token_pairs = []
        attention_to_A = []
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"  Sample {i}/{n_samples}...")
            
            A = np.random.randint(0, self.tokenizer.vocab_size)
            B = np.random.randint(0, self.tokenizer.vocab_size)
            
            try:
                input_ids = torch.tensor([[A, B]]).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True)
                    h3 = outputs.hidden_states[4][0, 1].float().cpu().numpy()
                    attn3 = outputs.attentions[3][0, :, 1, 0].mean().item()
                
                h3_outputs.append(h3)
                token_pairs.append((A, B))
                attention_to_A.append(attn3)
                
            except:
                continue
        
        h3_outputs = np.array(h3_outputs)
        attention_to_A = np.array(attention_to_A)
        
        print(f"  Collected {len(h3_outputs)} samples")
        print(f"  h3 shape: {h3_outputs.shape}")
        
        return h3_outputs, token_pairs, attention_to_A
    
    def analyze_clustering(self, h3_outputs: np.ndarray, n_clusters_list: List[int] = [10, 50, 100, 500]):
        """Test if layer 3 outputs cluster into types."""
        print(f"\n--- Clustering Analysis ---")
        
        # First, reduce dimensionality for visualization and clustering
        print(f"  Reducing dimensionality with PCA...")
        pca = PCA(n_components=50)
        h3_reduced = pca.fit_transform(h3_outputs)
        
        variance_explained = pca.explained_variance_ratio_.cumsum()
        print(f"    50 components explain {variance_explained[-1]*100:.1f}% variance")
        
        # Test different numbers of clusters
        for n_clusters in n_clusters_list:
            if n_clusters > len(h3_outputs):
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(h3_reduced)
            
            # Measure cluster quality
            # For each sample, compute distance to its centroid vs distance to nearest other centroid
            intra_cluster_dist = []
            inter_cluster_dist = []
            
            for i in range(len(h3_reduced)):
                own_centroid = kmeans.cluster_centers_[labels[i]]
                dist_to_own = np.linalg.norm(h3_reduced[i] - own_centroid)
                intra_cluster_dist.append(dist_to_own)
                
                # Distance to nearest other centroid
                min_other_dist = float('inf')
                for j in range(n_clusters):
                    if j != labels[i]:
                        dist = np.linalg.norm(h3_reduced[i] - kmeans.cluster_centers_[j])
                        min_other_dist = min(min_other_dist, dist)
                inter_cluster_dist.append(min_other_dist)
            
            silhouette = np.mean([(inter_cluster_dist[i] - intra_cluster_dist[i]) / 
                                  max(inter_cluster_dist[i], intra_cluster_dist[i])
                                  for i in range(len(h3_reduced))])
            
            print(f"    k={n_clusters}: silhouette={silhouette:.3f}")
        
        return pca, h3_reduced
    
    def test_centroid_accuracy(self, h3_outputs: np.ndarray, token_pairs: List, n_clusters: int = 100):
        """
        Test: If we use cluster centroids instead of actual h3, what's the accuracy?
        """
        print(f"\n--- Centroid Accuracy Test (k={n_clusters}) ---")
        
        # Cluster
        pca = PCA(n_components=50)
        h3_reduced = pca.fit_transform(h3_outputs)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(h3_reduced)
        
        # For each sample, compare:
        # 1. True token (from actual h3)
        # 2. Predicted token (from centroid)
        
        # We need to map centroids back to full dimension
        centroids_full = pca.inverse_transform(kmeans.cluster_centers_)
        
        correct = 0
        for i in range(len(h3_outputs)):
            true_token = np.argmax(self.lm_head @ h3_outputs[i])
            
            # Use centroid
            centroid = centroids_full[labels[i]]
            pred_token = np.argmax(self.lm_head @ centroid)
            
            if true_token == pred_token:
                correct += 1
        
        accuracy = correct / len(h3_outputs)
        print(f"  Centroid accuracy: {correct}/{len(h3_outputs)} = {accuracy*100:.1f}%")
        
        # But wait - this is using h3 directly with lm_head
        # The actual prediction requires running layers 4-27
        # Let's check if h3 alone predicts the final token
        
        print(f"\n  Note: This uses h3 directly with lm_head (not running layers 4-27)")
        
        return accuracy
    
    def analyze_attention_structure(self, h3_outputs: np.ndarray, attention_to_A: np.ndarray):
        """
        Analyze if attention patterns correlate with h3 structure.
        """
        print(f"\n--- Attention Structure Analysis ---")
        
        # Bin attention values
        bins = [0, 0.3, 0.5, 0.7, 1.0]
        bin_labels = ['low (0-0.3)', 'med-low (0.3-0.5)', 'med-high (0.5-0.7)', 'high (0.7-1.0)']
        
        for i in range(len(bins) - 1):
            mask = (attention_to_A >= bins[i]) & (attention_to_A < bins[i+1])
            count = mask.sum()
            
            if count > 1:
                h3_bin = h3_outputs[mask]
                
                # Compute pairwise cosine similarities within bin
                cos_sims = []
                for j in range(min(100, count)):
                    for k in range(j+1, min(100, count)):
                        cos = np.dot(h3_bin[j], h3_bin[k]) / (
                            np.linalg.norm(h3_bin[j]) * np.linalg.norm(h3_bin[k]) + 1e-10)
                        cos_sims.append(cos)
                
                if cos_sims:
                    print(f"    {bin_labels[i]}: n={count}, mean_cos={np.mean(cos_sims):.4f}")
        
        # Overall correlation between attention and h3 structure
        print(f"\n  Attention to A distribution:")
        print(f"    Mean: {attention_to_A.mean():.3f}")
        print(f"    Std: {attention_to_A.std():.3f}")
        print(f"    Min: {attention_to_A.min():.3f}, Max: {attention_to_A.max():.3f}")


def main():
    print("=" * 70)
    print("LAYER 3 CLUSTERING")
    print("=" * 70)
    print("""
Can we compress the "click" by clustering layer 3 outputs?

If layer 3 outputs cluster into types, we can:
1. Cache cluster centroids
2. Predict cluster from (A, B)
3. Use centroid instead of computing layer 3
""")
    
    analyzer = Layer3ClusteringAnalyzer()
    
    # 1. Collect layer 3 outputs
    h3_outputs, token_pairs, attention_to_A = analyzer.collect_layer3_outputs(n_samples=500)
    
    # 2. Analyze clustering
    pca, h3_reduced = analyzer.analyze_clustering(h3_outputs)
    
    # 3. Test centroid accuracy
    analyzer.test_centroid_accuracy(h3_outputs, token_pairs, n_clusters=100)
    
    # 4. Analyze attention structure
    analyzer.analyze_attention_structure(h3_outputs, attention_to_A)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
