"""Embed every paper in the DB and (optionally) cluster them into concepts.

Run:
    # embed only
    python scripts/run_embed_papers.py

    # embed + cluster + write concepts back to the DB
    python scripts/run_embed_papers.py --cluster kmeans
    python scripts/run_embed_papers.py --cluster hdbscan --min-cluster-size 5

Pipeline:
    1. sentence-transformers encodes title+abstract for each paper.
    2. Vectors are persisted into paper_embeddings.
    3. (--cluster) sklearn clusters the matrix; TF-IDF names each cluster;
       Concept + PaperConcept rows are written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xuanzhi.db import Store
from xuanzhi.nlp import Embedder, cluster_embeddings, derive_concepts_from_clusters
from xuanzhi.nlp.embeddings import EmbeddingMatrix, embed_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed + optionally cluster papers.")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "xuanzhi.db")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model id.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-embed every paper, not just the ones missing an embedding.",
    )
    parser.add_argument(
        "--cluster",
        choices=["kmeans", "hdbscan"],
        default=None,
        help="If set, cluster the embeddings and write Concept rows.",
    )
    parser.add_argument("--k", type=int, default=None, help="KMeans cluster count.")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = Store(args.db)
    total = store.count_papers()
    if total == 0:
        print("DB is empty — run scripts/run_arxiv_ingest.py first.")
        return

    embedder = Embedder(model_name=args.model)
    n = embed_corpus(store, embedder, only_missing=not args.all)
    print(f"Embedded {n} papers ({total} in DB) with {args.model}.")

    if args.cluster:
        matrix = EmbeddingMatrix.from_store(store, model=args.model)
        labels = cluster_embeddings(
            matrix,
            method=args.cluster,
            k=args.k,
            min_cluster_size=args.min_cluster_size,
        )
        concepts = derive_concepts_from_clusters(store, matrix, labels, persist=True)
        print(f"\nClustered into {len(concepts)} concepts via {args.cluster}:")
        for cluster_id, concept in sorted(concepts.items()):
            size = int((labels == cluster_id).sum())
            print(f"  [{cluster_id:2d}] ({size:3d} papers)  {concept.name}")


if __name__ == "__main__":
    main()
