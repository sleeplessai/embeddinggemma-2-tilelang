"""
Inference Demonstration for EmbeddingGemma-2.
=============================================
Usage:
    python inference/generate.py [--model_dir PATH] [--device cuda]
"""

import os
import sys
import argparse
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from model import EmbeddingGemma2
from ops import warmup_ops


def main():
    parser = argparse.ArgumentParser(description="EmbeddingGemma-2 TileLang Semantic Search")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to official EmbeddingGemma-2 directory (containing model.safetensors, config.json, tokenizer.json)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Target device (e.g. cuda, cpu)")
    parser.add_argument("--warmup", action="store_true", help="Pre-compile GPU kernels ahead of time")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    print("=" * 72)
    print(" EmbeddingGemma-2: High-Performance Semantic Representation")
    print("=" * 72)

    if args.warmup:
        warmup_ops()

    print(f"\n[1/3] Loading model checkpoint from: {args.model_dir} ...")
    model = EmbeddingGemma2.from_pretrained(args.model_dir, device=args.device)
    model.eval()

    queries = [
        "What causes the northern lights in polar skies?",
        "How do transformers use multi-head attention?",
    ]

    documents = [
        "The aurora borealis occurs when solar wind charged particles interact with Earth's magnetosphere.",
        "Multi-head attention projects queries, keys, and values into subspaces to model diverse contextual relationships.",
        "Deep neural networks require large-scale compute clusters for training and gradient backpropagation.",
        "Traditional Neapolitan pizza requires high-temperature wood-fired ovens, San Marzano tomatoes, and mozzarella.",
    ]

    print(f"\n[2/3] Encoding {len(queries)} search queries & {len(documents)} documents...")
    query_embs = model.encode(queries, prompt_name="SearchQuery")
    doc_embs = model.encode(documents, prompt_name="Document")

    print(f"      Query Embeddings:    {list(query_embs.shape)}")
    print(f"      Document Embeddings: {list(doc_embs.shape)}")

    print("\n[3/3] Pairwise Cosine Similarity & Top Matches:")
    sim_matrix = torch.matmul(query_embs.float(), doc_embs.float().T)

    for q_idx, q_text in enumerate(queries):
        print(f"\nQuery [{q_idx + 1}]: \"{q_text}\"")
        scores = sim_matrix[q_idx]
        ranked_indices = torch.argsort(scores, descending=True)
        for rank, doc_idx in enumerate(ranked_indices):
            print(f"  Rank {rank + 1} | Score: {scores[doc_idx].item():.4f} | \"{documents[doc_idx]}\"")

    print("\nMatryoshka Representation Learning (MRL) Truncation:")
    for dim in [768, 512, 256, 128]:
        q_trunc = model.encode(queries[:1], prompt_name="SearchQuery", truncate_dim=dim)
        d_trunc = model.encode(documents[:2], prompt_name="Document", truncate_dim=dim)
        score = torch.matmul(q_trunc.float(), d_trunc.float().T)[0]
        print(f"  Dim {dim:3d}d: Match Score = {score[0].item():.4f} | Other Score = {score[1].item():.4f}")

    print("\n" + "=" * 72)
    print("Inference completed successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()
