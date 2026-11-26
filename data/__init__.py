"""The datasets of the three problems: their generators, and the split inputs that every method reads."""

PROBLEMS = ("ba", "kp", "bm")  # budget allocation, knapsack, bipartite matching
SPLITS = {"ba": (320, 400, 600), "kp": (320, 400, 600), "bm": (90, 105, 135)}  # row where TRAIN, VALID and TEST end
KP_CAPACITY = 30
BM_NODES = 10  # nodes on each side of the bipartite graph
