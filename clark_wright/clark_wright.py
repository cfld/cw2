#!/usr/bin/env python

"""
    clark_wright/clark_wright.py
"""

import numpy as np

# --
# Helpers

def make_DI(dist, n_close):
    """ make neighbor shortlists """
    mask = 100 * dist.max() * np.eye(dist.shape[0])
    tmp  = dist + mask
    
    D = -1 * np.sort(-1 * tmp, axis=-1, kind='stable')[:,::-1][:,:n_close]
    I = np.argsort(-1 * tmp, axis=-1, kind='stable')[:,::-1][:,:n_close]
    return D, I

def DI_to_edges(D, I, D_depot):
    """ convert neighbor shortlists to edges for CW """
    n_customers = D.shape[0]
    n_neighbors = D.shape[1]
    
    srcs = np.repeat(np.arange(n_customers), n_neighbors)
    dsts = I.ravel()
    
    savs = D_depot.reshape(-1, 1) + D_depot[I] - D
    savs = savs.ravel()
    savs[savs < 0.1] = 0.1
    
    dists = D.ravel()
    
    p = np.argsort(-savs, kind='stable')
    srcs, dsts, savs, dists = srcs[p], dsts[p], savs[p], dists[p]
    
    srcs = list(srcs.astype(int))
    dsts = list(dsts.astype(int))
    
    edges = {}
    for src, dst, d in zip(srcs, dsts, dists):
        edges[(src, dst)] = d
    
    return edges

# --
# Clark-Wright savings algorithm

class __CW:
    def __init__(self, D, I, D_depot, demand, cap):
        self.edges       = DI_to_edges(D, I, D_depot)
        self.D_depot     = D_depot
        self.demand      = demand
        self.cap         = cap
        
        self.n_customers = I.shape[0]
        self.depot_id    = I.shape[0]
        
        self.visited    = set([])
        self.boundary   = set([])
        self.routes     = {}
        self.node2route = {}
        self.route_idx  = 0
    
    def _get_dist(self, a, b):
        if (a, b) in self.edges:
            return self.edges[(a, b)]
        elif (b, a) in self.edges:
            return self.edges[(b, a)]
        else:
            raise Exception
    
    def _new_route(self, src, dst):
        load = self.demand[src] + self.demand[dst]
        cost = self.D_depot[src] + self.edges[(src, dst)] + self.D_depot[dst]
        
        if load > self.cap:
            return
        
        self.visited.add(src)
        self.visited.add(dst)
        self.boundary.add(src)
        self.boundary.add(dst)
        
        self.node2route[src] = self.route_idx
        self.node2route[dst] = self.route_idx
        
        self.routes[self.route_idx] = {
            "idx"   : self.route_idx,
            "nodes" : [src, dst],
            "load"  : load,
            "cost"  : cost,
        }
        
        self.route_idx += 1
        
    def _extend_route(self, a, b):
        r = self.routes[self.node2route[a]]
        
        new_load = r['load'] + self.demand[b]
        new_cost = r['cost'] + self._get_dist(a, b) + self.D_depot[b] - self.D_depot[a]
        
        if new_load > self.cap:
            return
            
        self.visited.add(b)
        self.boundary.remove(a)
        self.boundary.add(b)
        
        if r['nodes'][0] == a:
            r['nodes'].insert(0, b)
        elif r['nodes'][-1] == a:
            r['nodes'].append(b)
        else:
            raise Exception('not in right position')
            
        r['load'] = new_load
        r['cost'] = new_cost
        
        self.node2route[b] = r['idx']
        
    def _merge_route(self, src, dst):
        r_src = self.routes[self.node2route[src]]
        r_dst = self.routes[self.node2route[dst]]
        
        new_load = r_src['load'] + r_dst['load']
        new_cost = r_src['cost'] + r_dst['cost'] + self.edges[(src, dst)] - self.D_depot[src] - self.D_depot[dst]
        
        if new_load > self.cap:
            return
            
        self.boundary.remove(src)
        self.boundary.remove(dst)
        
        # reverse direction to fit
        if r_src['nodes'][-1] != src:
            r_src['nodes'] = r_src['nodes'][::-1]
        
        if r_dst['nodes'][0] != dst:
            r_dst['nodes'] = r_dst['nodes'][::-1]
        
        del self.routes[self.node2route[src]]
        del self.routes[self.node2route[dst]]
        
        r = {
            "idx"   : self.route_idx,
            "nodes" : r_src['nodes'] + r_dst['nodes'],
            "load"  : new_load,
            "cost"  : new_cost,
        }
        for n in r['nodes']:
            self.node2route[n] = self.route_idx
        
        self.routes[self.route_idx] = r
        self.route_idx += 1
    
    def _fix_unvisited(self):
        # fix customers that haven't been visited
        for n in range(self.n_customers):
            if n not in self.visited:
                self.routes[self.route_idx] = {
                    "idx"    : self.route_idx,
                    "nodes"  : [n],
                    "load"   : self.demand[n],
                    "cost"   : 2 * self.D_depot[n],
                }
                self.route_idx += 1
    
    def run(self):
        for (src, dst) in self.edges.keys():
            
            src_visited  = src in self.visited
            dst_visited  = dst in self.visited
            src_boundary = src in self.boundary
            dst_boundary = dst in self.boundary
            
            if src_visited and not src_boundary:
                pass
            
            elif dst_visited and not dst_boundary:
                pass
            
            elif not src_visited and not dst_visited:
                self._new_route(src, dst)
            
            elif src_boundary and not dst_visited:
                self._extend_route(src, dst)
            
            elif dst_boundary and not src_visited:
                self._extend_route(dst, src)
            
            elif src_boundary and dst_boundary and (self.node2route[src] != self.node2route[dst]):
                self._merge_route(src, dst)
            
            else:
                pass
        
        self._fix_unvisited()
        
        routes = [r['nodes'] for r in self.routes.values()]
        return routes

# --
# High-level interface

def clark_wright(dist, demand, cap, depot_id, n_close):
    """
        Run Clark-Wright savings algorithm
        
        dist     : distance matrix
        demand   : demand vector
        cap      : maximum vehicle capacity
        depot_id : depot_id
        n_close  : number of neighbors for CW shortlists
        
    """
    idxs     = np.arange(dist.shape[0])
    node_sel = np.setdiff1d(idxs, depot_id)
    
    # distance from nodes to depot
    dist_depot = dist[depot_id][node_sel]
    
    # shortlisted distance from node to nodes
    dist_nodes = dist[node_sel][:,node_sel]
    dist_neib, idx_neib = make_DI(dist_nodes, n_close)
    
    return __CW(dist_neib, idx_neib, dist_depot, demand, cap).run()
