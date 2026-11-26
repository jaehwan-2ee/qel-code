"""BM (bipartite matching on Cora) dataset of one data seed, from `cora.cites` and `cora.content`.

The citation graph is cut into 135 parts by the METIS partition in data/bm/precomputed, and the parts are resized to
20 papers each: larger parts give up their papers of lowest degree, smaller parts take them. An instance is one part
with its papers split at random into 10 left and 10 right nodes; the true value of a pair is 1 when one paper cites
the other, and its features are the 1433 binary word features of both papers. The parts are assigned to TRAIN, VALID
and TEST by a seeded permutation. This follows the construction of OpenPTO's BipartiteMatching (27 parts of 100
papers there) at commit f885426f.
"""

from pathlib import Path

import networkx as nx
import numpy as np

from data import BM_NODES, SPLITS
from data.common import check_raw_file

WORDS = 1433
RAW = {
    "cora.cites": (69_928, "ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e"),
    "cora.content": (7_823_427, "b8ebed0fd45e53995abafd1b70cf6a9fcc8f86abc5826d69b97fe6ca46e9c8ee"),
}
PARTITION = Path(__file__).parent / "bm/precomputed/cora_cites_metis.txt.part.135"
PARTITION_SIZE, PARTITION_SHA256 = 8_626, "fd3bf37686c2eccc921f5346097467b4af0ec5697954f461a3864f0349674ef5"


def _citation_graph(path):
    """The graph with its nodes numbered in order of first appearance, and the paper id of every node."""
    graph = nx.Graph()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            graph.add_edge(*map(int, line.split()))
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, label_attribute="paper")
    return graph, {int(node): int(data["paper"]) for node, data in graph.nodes(data=True)}


def _parts(graph, assignments):
    """The parts of the METIS partition, resized to 2 * BM_NODES papers each."""
    size = 2 * BM_NODES
    nodes_of = [list(np.flatnonzero(assignments == part)) for part in range(SPLITS["bm"][2])]
    pooled = []
    for nodes in nodes_of:
        part = graph.subgraph(nodes)
        if len(part) > size:
            removed = sorted(part.nodes, key=lambda node: (part.degree(node), int(node)))[: len(part) - size]
            nodes[:] = [node for node in nodes if node not in removed]
            pooled.extend(int(node) for node in removed)
    for nodes in nodes_of:
        needed = size - len(nodes)
        if needed > 0:
            nodes.extend(pooled[:needed])
            del pooled[:needed]
    return [graph.subgraph(nodes).copy() for nodes in nodes_of]


def generate(raw, seed):
    """X (135, 100, 2866) float32 and y (135, 100) float32. Pair 10 i + j of an instance is (left node i, right node j)
    with the features [words of the left paper | words of the right paper]. Rows TRAIN 0:90, VALID 90:105,
    TEST 105:135."""
    for name, (size, digest) in RAW.items():
        check_raw_file(raw / "bm" / name, size, digest)
    if not PARTITION.is_file():
        raise SystemExit(f"{PARTITION} is missing; it is a file of this repository")
    check_raw_file(PARTITION, PARTITION_SIZE, PARTITION_SHA256)
    graph, paper = _citation_graph(raw / "bm/cora.cites")
    parts = _parts(graph, np.loadtxt(PARTITION))
    content = np.loadtxt(raw / "bm/cora.content", dtype=np.int64, usecols=range(WORDS + 1))  # the label is not read
    words = {int(row[0]): row[1:].astype(np.float32) for row in content}
    rng = np.random.RandomState(seed)
    sides = []
    for part in parts:
        nodes = [int(node) for node in rng.permutation(sorted(part.nodes)).tolist()]
        sides.append((nodes[:BM_NODES], nodes[BM_NODES:]))
    order = np.random.RandomState(seed).permutation(len(parts)).tolist()  # rows: TRAIN, then VALID, then TEST
    X = np.empty((len(parts), BM_NODES**2, 2 * WORDS), np.float32)
    y = np.empty((len(parts), BM_NODES**2), np.float32)
    for row, part in enumerate(order):
        left, right = sides[part]
        left_words, right_words = (np.stack([words[paper[node]] for node in side]) for side in (left, right))
        X[row] = np.concatenate((np.repeat(left_words, BM_NODES, axis=0), np.tile(right_words, (BM_NODES, 1))), axis=1)
        y[row] = [parts[part].has_edge(i, j) for i in left for j in right]
    return {"X": X, "y": y}, {"seed": seed, "source_parts": order}
