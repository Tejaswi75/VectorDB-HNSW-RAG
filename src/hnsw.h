#pragma once
// HNSW — Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016)
#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "distance.h"

// =====================================================================
//  HNSW — Hierarchical Navigable Small World
// =====================================================================

class HNSW {
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int    M, M0, ef_build;
    float  mL;
    int    topLayer = -1;
    int    entryPt  = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return (int)std::floor(-std::log(u(rng)) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if ((int)found.size() < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    // Neighbour-selection heuristic (Algorithm 4 in the HNSW paper).
    // A candidate is kept only if it is closer to the base point than to every
    // neighbour already chosen. This keeps "bridge" edges between clusters
    // instead of linking only to the nearest points of one dense region, which
    // otherwise splits the graph into islands and hurts recall.
    // Slots left over are filled with the closest pruned candidates.
    std::vector<int> selectNbrs(const std::vector<std::pair<float,int>>& cands,
                                int maxM, DistFn dist)
    {
        std::vector<int> r, pruned;
        for (auto& [dq, cid] : cands) {          // cands are sorted by distance
            if ((int)r.size() >= maxM) break;
            bool good = true;
            for (int sid : r)
                if (dist(G[cid].item.emb, G[sid].item.emb) < dq) { good = false; break; }
            if (good) r.push_back(cid); else pruned.push_back(cid);
        }
        for (int i = 0; (int)r.size() < maxM && i < (int)pruned.size(); i++)
            r.push_back(pruned[i]);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2*m), ef_build(efBuild),
          mL(1.0f / std::log((float)m)), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G[id]   = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; lc--) {
            auto W   = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM, dist);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if ((int)G[nid].nbrs.size() <= lc) G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if ((int)conn.size() > maxM) {
                    std::vector<std::pair<float,int>> ds;
                    for (int c : conn) if (G.count(c))
                        ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn = selectNbrs(ds, maxM, dist);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    void remove(int id) {
        if (!G.count(id)) return;
        for (auto& [nid, nd] : G)
            for (auto& layer : nd.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
        if (entryPt == id) {
            entryPt = -1;
            for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }
        }
        G.erase(id);
    }

    struct GraphInfo {
        int topLayer, nodeCount;
        std::vector<int> nodesPerLayer, edgesPerLayer;
        struct NV { int id; std::string metadata, category; int maxLyr; };
        struct EV { int src, dst, lyr; };
        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo() {
        GraphInfo gi;
        gi.topLayer  = topLayer;
        gi.nodeCount = (int)G.size();
        int maxL = std::max(topLayer + 1, 1);
        gi.nodesPerLayer.assign(maxL, 0);
        gi.edgesPerLayer.assign(maxL, 0);
        for (auto& [id, nd] : G) {
            gi.nodes.push_back({id, nd.item.metadata, nd.item.category, nd.maxLyr});
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < (int)nd.nbrs.size())
                    for (int nid : nd.nbrs[lc])
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.push_back({id, nid, lc});
                        }
            }
        }
        return gi;
    }

    size_t size() const { return G.size(); }
};
