"""Stage 4 — Conflict Graph Construction.

Clusters claims by embedding similarity, classifies intra-cluster pairs
via the LLM, and builds a typed NetworkX directed graph.
"""

from __future__ import annotations

import json
import re
import uuid
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import structlog

from src.schemas import Claim, ConflictEdge, RelationType
from src.utils.llm import LLMClient
from src.utils.prompts import load_prompt
from src.vectorstore.embeddings import EmbeddingModel

log = structlog.get_logger()

_VALID_RELATIONS = {r.value for r in RelationType}


class ConflictGraphBuilder:
    """Builds a claim conflict graph using embedding clustering + LLM classification."""

    def __init__(
        self,
        llm: LLMClient,
        embedder: EmbeddingModel,
        similarity_threshold: float = 0.6,
    ) -> None:
        """Initialise the builder.

        Args:
            llm: LLM client for pairwise claim classification.
            embedder: Embedding model for clustering claims.
            similarity_threshold: Minimum cosine similarity for two claims
                to be considered in the same cluster and worth comparing.
        """
        self._llm = llm
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self, claims: list[Claim]) -> nx.DiGraph:
        """Build a conflict graph from a list of claims.

        1. Embed all claims and cluster by cosine similarity.
        2. For each intra-cluster pair, call the LLM to classify.
        3. Filter out UNRELATED edges.
        4. Return a NetworkX DiGraph.

        Args:
            claims: Extracted claims to analyse.

        Returns:
            Directed graph with claims as nodes and typed edges.
        """
        if len(claims) < 2:
            return self._empty_graph(claims)

        pairs = self._cluster_and_pair(claims)
        log.info("conflict_graph_pairs", total_claims=len(claims), pairs_to_classify=len(pairs))

        edges: list[ConflictEdge] = []
        for claim_a, claim_b in pairs:
            edge = self._classify_pair(claim_a, claim_b)
            if edge is not None and edge.relation != "UNRELATED":
                edges.append(edge)

        return self._assemble_graph(claims, edges)

    @staticmethod
    def get_conflicts(graph: nx.DiGraph) -> list[ConflictEdge]:
        """Return only CONTRADICTS and SUPERSEDES edges from a built graph.

        Args:
            graph: A conflict graph built by :meth:`build_graph`.

        Returns:
            List of ConflictEdge objects with relation CONTRADICTS or SUPERSEDES.
        """
        result: list[ConflictEdge] = []
        for _, _, data in graph.edges(data=True):
            edge: ConflictEdge = data["edge"]
            if edge.relation in (RelationType.CONTRADICTS, RelationType.SUPERSEDES):
                result.append(edge)
        return result

    @staticmethod
    def visualize_graph(graph: nx.DiGraph, output_path: str | Path) -> None:
        """Render the conflict graph as an interactive HTML file using pyvis.

        Args:
            graph: A conflict graph built by :meth:`build_graph`.
            output_path: File path for the generated HTML (e.g. ``"graph.html"``).
        """
        from pyvis.network import Network

        net = Network(directed=True, height="700px", width="100%")

        color_map = {
            RelationType.SUPPORTS: "#4CAF50",
            RelationType.CONTRADICTS: "#F44336",
            RelationType.SUPERSEDES: "#FF9800",
        }

        for node_id, data in graph.nodes(data=True):
            claim: Claim = data["claim"]
            label = claim.text[:80] + ("..." if len(claim.text) > 80 else "")
            net.add_node(node_id, label=label, title=claim.text)

        for u, v, data in graph.edges(data=True):
            edge: ConflictEdge = data["edge"]
            color = color_map.get(edge.relation, "#9E9E9E")
            net.add_edge(
                u, v,
                label=edge.relation.value,
                color=color,
                title=edge.reasoning,
            )

        net.write_html(str(output_path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster_and_pair(
        self, claims: list[Claim],
    ) -> list[tuple[Claim, Claim]]:
        """Embed claims, compute pairwise cosine sim, return above-threshold pairs.

        Args:
            claims: Claims to cluster.

        Returns:
            List of (claim_a, claim_b) pairs above the similarity threshold.
        """
        texts = [c.text for c in claims]
        embeddings = np.array(self._embedder.embed_batch(texts))

        # Cosine similarity matrix (vectors are already L2-normalised)
        sim_matrix = embeddings @ embeddings.T

        pairs: list[tuple[Claim, Claim]] = []
        n = len(claims)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self._similarity_threshold:
                    pairs.append((claims[i], claims[j]))
        return pairs

    def _classify_pair(self, claim_a: Claim, claim_b: Claim) -> ConflictEdge | None:
        """Call the LLM to classify the relationship between two claims.

        Returns ``None`` when the LLM response cannot be parsed.

        Args:
            claim_a: Source claim.
            claim_b: Target claim.

        Returns:
            A ConflictEdge, or None on parse failure.
        """
        prompt = load_prompt(
            "conflict_classification",
            claim_a=claim_a.text,
            claim_b=claim_b.text,
        )

        raw = self._llm.invoke(prompt)
        return self._parse_classification(raw, claim_a.claim_id, claim_b.claim_id)

    def _parse_classification(
        self, raw: str, source_id: str, target_id: str,
    ) -> ConflictEdge | None:
        """Parse the LLM classification response into a ConflictEdge.

        Args:
            raw: Raw LLM output (should be a JSON object).
            source_id: claim_id of the source claim.
            target_id: claim_id of the target claim.

        Returns:
            Parsed ConflictEdge, or None on failure.
        """
        try:
            data = _extract_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            log.warning("conflict_classification_parse_failed", raw_head=raw[:200])
            return None

        relation_raw = data.get("relation", "UNRELATED").upper()
        if relation_raw not in _VALID_RELATIONS:
            relation_raw = "UNRELATED"

        confidence = data.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.5

        reasoning = str(data.get("reasoning", "")).strip() or "No reasoning provided."

        return ConflictEdge(
            source_claim_id=source_id,
            target_claim_id=target_id,
            relation=RelationType(relation_raw),
            confidence=confidence,
            reasoning=reasoning,
        )

    @staticmethod
    def _assemble_graph(
        claims: list[Claim], edges: list[ConflictEdge],
    ) -> nx.DiGraph:
        """Build a NetworkX DiGraph from claims and edges.

        Args:
            claims: All claims (graph nodes).
            edges: Classified edges (non-UNRELATED).

        Returns:
            Populated DiGraph.
        """
        g = nx.DiGraph()
        for c in claims:
            g.add_node(c.claim_id, claim=c)
        for e in edges:
            g.add_edge(
                e.source_claim_id,
                e.target_claim_id,
                edge=e,
                relation=e.relation.value,
            )
        return g

    @staticmethod
    def _empty_graph(claims: list[Claim]) -> nx.DiGraph:
        """Return a graph with nodes only (0 or 1 claims)."""
        g = nx.DiGraph()
        for c in claims:
            g.add_node(c.claim_id, claim=c)
        return g


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict:
    """Extract and parse the first JSON object ``{...}`` from *text*.

    Handles markdown fences and leading/trailing prose.

    Args:
        text: Raw LLM output.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If no JSON object is found.
        json.JSONDecodeError: If the object is malformed.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])

    raise ValueError("Unbalanced braces in LLM output")
