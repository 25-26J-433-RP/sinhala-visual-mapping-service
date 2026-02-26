"""
Graph Constraints Post-Processor for Sinhala Concept Maps.

Applies four correctness rules to the raw nodes+edges produced by the
intelligent generator AFTER all edges have been assembled but BEFORE
positions are assigned:

  1. Merge near-duplicate nodes
     – Nodes whose labels share > MERGE_THRESHOLD trigram overlap are
       collapsed into one (keeping the higher-importance node) and all
       edges are re-wired to the survivor.

  2. Remove cycles in hierarchy
     – Edges typed 'hierarchy', 'is-a', or 'part-of' must form a DAG.
       Detected back-edges are removed, preferring to cut the edge with
       the lowest (weight x confidence) score so the weakest link in
       every cycle is pruned first.

  3. Limit parents per node
     – Each node may receive at most MAX_HIERARCHY_PARENTS incoming
       hierarchy edges.  When the limit is exceeded the weakest edges
       (lowest weight, then confidence) are removed.

  4. Penalise cross-cluster weak edges
     – Edges of 'soft' types (related-to, semantic, proximity,
       conjunction) that cross cluster boundaries AND whose confidence
       falls below CROSS_CLUSTER_MIN_CONF are dropped.  Structural
       types (hierarchy, is-a, part-of, cause-effect) are never pruned
       by this rule.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
MERGE_THRESHOLD: float = 0.82        # trigram-Jaccard above this → merge
MAX_HIERARCHY_PARENTS: int = 2       # max incoming hierarchy edges per node
CROSS_CLUSTER_MIN_CONF: float = 0.58 # stricter minimum confidence for cross-cluster soft edges

_HIERARCHY_TYPES: Set[str] = {'hierarchy', 'is-a', 'part-of'}
_SOFT_TYPES: Set[str] = {
    'related-to', 'semantic', 'semantic_cross',
    'proximity', 'conjunction', 'close',
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GraphConstraints:
    """
    Apply all four post-processing constraints to a (nodes, edges) graph.

    Usage::

        gc = GraphConstraints()
        nodes, edges = gc.apply(nodes, edges, cluster_map)

    Parameters
    ----------
    nodes        : list of node dicts  (keys: id, label, importance, ...)
    edges        : list of edge dicts  (keys: id, source, target, type,
                                        weight, confidence, ...)
    cluster_map  : dict mapping node-label → cluster-index  (may be empty)
    """

    def apply(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        cluster_map: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cluster_map = cluster_map or {}

        n_before = len(nodes)
        e_before = len(edges)

        nodes, edges = self._merge_near_duplicates(nodes, edges)
        edges       = self._remove_hierarchy_cycles(nodes, edges)
        edges       = self._limit_parents(nodes, edges)
        edges       = self._penalise_cross_cluster(nodes, edges, cluster_map)

        logger.info(
            "GraphConstraints: nodes %d→%d  edges %d→%d",
            n_before, len(nodes), e_before, len(edges),
        )
        return nodes, edges

    # =========================================================================
    # 1. Merge near-duplicate nodes
    # =========================================================================

    def _merge_near_duplicates(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Collapse nodes whose labels are very similar (trigram-Jaccard ≥
        MERGE_THRESHOLD).  The survivor is always the node with higher
        importance; all edges referencing the duplicate are re-wired.
        """
        if len(nodes) < 2:
            return nodes, edges

        # Build a union-find structure so we can collapse chains of similar
        # nodes without bias from iteration order.
        parent: Dict[str, str] = {n['id']: n['id'] for n in nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        def union(a: str, b: str, nodes_by_id: Dict[str, Dict]) -> None:
            """Keep the node with higher importance as the root."""
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            imp_a = nodes_by_id[ra].get('importance', 0)
            imp_b = nodes_by_id[rb].get('importance', 0)
            # Root should be the survivor (higher importance)
            if imp_a >= imp_b:
                parent[rb] = ra
            else:
                parent[ra] = rb

        nodes_by_id = {n['id']: n for n in nodes}
        id_list = [n['id'] for n in nodes]

        # Compare all pairs – O(n²) but n is small (≤ 50 nodes by design)
        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                ni = nodes_by_id[id_list[i]]
                nj = nodes_by_id[id_list[j]]
                # Skip if already in the same component
                if find(id_list[i]) == find(id_list[j]):
                    continue
                sim = _trigram_jaccard(ni.get('label', ''), nj.get('label', ''))
                if sim >= MERGE_THRESHOLD:
                    union(id_list[i], id_list[j], nodes_by_id)
                    logger.debug(
                        "Merging near-duplicate '%s' ↔ '%s' (sim=%.2f)",
                        ni['label'], nj['label'], sim,
                    )

        # Build survivor set
        survivors: Dict[str, str] = {nid: find(nid) for nid in id_list}
        survivor_ids: Set[str] = set(survivors.values())

        # Keep only survivor nodes
        new_nodes = [n for n in nodes if n['id'] in survivor_ids]

        # Re-wire edges: replace any endpoint that points to a merged node
        def remap(nid: str) -> str:
            return survivors.get(nid, nid)

        new_edges: List[Dict[str, Any]] = []
        seen_edge_pairs: Set[Tuple[str, str, str]] = set()
        for e in edges:
            src = remap(e['source'])
            tgt = remap(e['target'])
            if src == tgt:            # self-loop from merge — discard
                continue
            key = (src, tgt, e.get('type', ''))
            if key in seen_edge_pairs:
                continue
            seen_edge_pairs.add(key)
            new_edges.append({**e, 'source': src, 'target': tgt})

        removed = len(nodes) - len(new_nodes)
        if removed:
            logger.info("Merged %d near-duplicate node(s)", removed)
        return new_nodes, new_edges

    # =========================================================================
    # 2. Remove cycles in hierarchy
    # =========================================================================

    def _remove_hierarchy_cycles(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Enforce that hierarchy/is-a/part-of edges form a DAG.

        Algorithm
        ---------
        Sort hierarchy edges by descending strength (weight × confidence).
        Greedily add edges that do not close a cycle (union-find reachability).
        Drop any edge that would create a cycle.
        Non-hierarchy edges pass through untouched.
        """
        hier_edges = [e for e in edges if e.get('type') in _HIERARCHY_TYPES]
        other_edges = [e for e in edges if e.get('type') not in _HIERARCHY_TYPES]

        if not hier_edges:
            return edges

        # Sort strongest first so cycle-breaking removes the weakest edge
        def _strength(e: Dict) -> float:
            return e.get('weight', 1) * e.get('confidence', 0.5)

        hier_edges_sorted = sorted(hier_edges, key=_strength, reverse=True)

        # Union-Find for reachability (undirected for cycle detection)
        nids = {n['id'] for n in nodes}
        uf_parent: Dict[str, str] = {nid: nid for nid in nids}

        def uf_find(x: str) -> str:
            while uf_parent.get(x, x) != x:
                uf_parent[x] = uf_parent.get(uf_parent[x], uf_parent[x])
                x = uf_parent[x]
            return x

        def uf_connected(a: str, b: str) -> bool:
            return uf_find(a) == uf_find(b)

        def uf_union(a: str, b: str) -> None:
            ra, rb = uf_find(a), uf_find(b)
            if ra != rb:
                uf_parent[rb] = ra

        accepted: List[Dict[str, Any]] = []
        removed = 0
        for e in hier_edges_sorted:
            src, tgt = e['source'], e['target']
            if uf_connected(src, tgt):
                # Adding this edge would close a cycle — reject
                logger.debug(
                    "Cycle removed: %s -[%s]-> %s", src, e.get('type'), tgt
                )
                removed += 1
            else:
                uf_union(src, tgt)
                accepted.append(e)

        if removed:
            logger.info("Removed %d cycle-forming hierarchy edge(s)", removed)
        return accepted + other_edges

    # =========================================================================
    # 3. Limit parents per node
    # =========================================================================

    def _limit_parents(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Each node may have at most MAX_HIERARCHY_PARENTS incoming hierarchy
        edges.  When the limit is exceeded the weakest edges are pruned
        (lowest weight, then confidence as tiebreaker).
        """
        from collections import defaultdict

        # Gather incoming hierarchy edges per target node
        incoming: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        other_edges: List[Dict[str, Any]] = []

        for e in edges:
            if e.get('type') in _HIERARCHY_TYPES:
                incoming[e['target']].append(e)
            else:
                other_edges.append(e)

        retained: List[Dict[str, Any]] = []
        total_removed = 0
        for tgt, in_edges in incoming.items():
            if len(in_edges) <= MAX_HIERARCHY_PARENTS:
                retained.extend(in_edges)
            else:
                # Sort by strength descending, keep top-k
                in_edges.sort(
                    key=lambda e: (e.get('weight', 1), e.get('confidence', 0.5)),
                    reverse=True,
                )
                retained.extend(in_edges[:MAX_HIERARCHY_PARENTS])
                dropped = len(in_edges) - MAX_HIERARCHY_PARENTS
                total_removed += dropped
                logger.debug(
                    "Node %s: dropped %d excess parent edge(s)", tgt, dropped
                )

        if total_removed:
            logger.info("Pruned %d excess parent edge(s)", total_removed)
        return retained + other_edges

    # =========================================================================
    # 4. Penalise cross-cluster weak edges
    # =========================================================================

    def _penalise_cross_cluster(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        cluster_map: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Drop 'soft' relationship edges (related-to, semantic, proximity, …)
        that cross cluster boundaries when their confidence is below
        CROSS_CLUSTER_MIN_CONF.

        Structural edges (hierarchy, is-a, part-of, cause-effect) are never
        removed by this rule — only soft co-occurrence/proximity edges.
        """
        if not cluster_map:
            return edges

        # Build id → cluster-index map using node labels as bridge
        label_to_cluster: Dict[str, int] = cluster_map  # label → int
        id_to_label: Dict[str, str] = {
            n['id']: n.get('label', n.get('source_text', ''))
            for n in nodes
        }
        id_to_cluster: Dict[str, int] = {
            nid: label_to_cluster[lbl]
            for nid, lbl in id_to_label.items()
            if lbl in label_to_cluster
        }

        id_to_level: Dict[str, int] = {n['id']: int(n.get('level', 0)) for n in nodes}

        retained: List[Dict[str, Any]] = []
        removed = 0
        for e in edges:
            etype = e.get('type', '')
            if etype not in _SOFT_TYPES:
                retained.append(e)
                continue

            c_src = id_to_cluster.get(e['source'])
            c_tgt = id_to_cluster.get(e['target'])

            # If cluster info is unavailable for either node, keep the edge
            if c_src is None or c_tgt is None:
                retained.append(e)
                continue

            if c_src != c_tgt:
                # Cross-cluster: enforce the confidence floor
                conf = e.get('confidence', 0.5)
                level_gap = abs(id_to_level.get(e['source'], 0) - id_to_level.get(e['target'], 0))
                effective_floor = CROSS_CLUSTER_MIN_CONF + (0.08 if level_gap >= 2 else 0.0)
                if conf < effective_floor:
                    logger.debug(
                        "Cross-cluster '%s' edge dropped (conf=%.2f < %.2f, level_gap=%d)",
                        etype, conf, effective_floor, level_gap,
                    )
                    removed += 1
                    continue

            retained.append(e)

        if removed:
            logger.info("Dropped %d weak cross-cluster edge(s)", removed)
        return retained


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> Set[str]:
    """Return the set of character 3-grams in *text* (lowercased)."""
    t = text.lower().strip()
    if len(t) < 3:
        return {t} if t else set()
    return {t[i: i + 3] for i in range(len(t) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    """Character trigram Jaccard similarity in [0, 1]."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb:
        return 1.0
    denom = len(ta | tb)
    return len(ta & tb) / denom if denom else 0.0
