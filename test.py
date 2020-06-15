
import argparse
import numpy as np
from tsplib95 import parser
from scipy.spatial.distance import pdist, squareform

from clark_wright.clark_wright import clark_wright

def load_problem(inpath, perturb=False):
    prob = parser.parse(open(inpath).read())
    
    cap         = prob['CAPACITY']
    n_customers = prob['DIMENSION'] - 1
    xy          = np.array(list(prob['NODE_COORD_SECTION'].values()))
    demand      = np.array(list(prob['DEMAND_SECTION'].values()))
    
    depot_id    = n_customers
    
    # COMPATIBILITY: Perturb locations to avoid ties
    if perturb:
        xy = xy + np.arange(xy.shape[0]).reshape(-1, 1) / xy.shape[0]
    
    # Move depot to end
    xy     = np.row_stack([xy[1:], xy[0]])
    demand = np.hstack([demand[1:], demand[0]])
    
    assert demand[depot_id] == 0
    
    return (
        cap,
        n_customers,
        depot_id,
        xy,
        demand,
    )

# --
# IO

n_close = 20
inpath  = '/Users/bjohnson/projects/fyre/VRPXXL/Leuven1.txt'

cap, n_customers, depot_id, xy, demand = load_problem(inpath)

# --
# Compute distance matrix

dist = squareform(pdist(xy))

# --
# Run Clark-Wright

init_routes = clark_wright(
    dist     = dist, 
    demand   = demand, 
    cap      = cap, 
    depot_id = depot_id, 
    n_close  = n_close,
)

# --
# Compute cost

total_cost = 0
for route in init_routes:
    cost = (
        + dist[depot_id, route[0]]
        + dist[(route[:-1], route[1:])].sum()
        + dist[route[-1], depot_id]
    )
    total_cost += cost

print(f'total_cost={total_cost}')