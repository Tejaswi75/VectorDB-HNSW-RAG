// Benchmark: HNSW vs KD-Tree vs Brute Force
//
// Measures index build time, average query latency and recall@k
// (against exact brute-force results) on synthetic clustered data.
//
// Build:  g++ -std=c++17 -O2 bench/benchmark.cpp -o benchmark
// Run:    ./benchmark [N] [dims...]        e.g.  ./benchmark 10000 16 128 768
//
// Euclidean distance is used throughout: KD-Tree pruning is only exact for
// L_p metrics, so this keeps the comparison fair.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../src/distance.h"
#include "../src/brute_force.h"
#include "../src/kd_tree.h"
#include "../src/hnsw.h"

using Clock = std::chrono::high_resolution_clock;

static double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Gaussian blobs around random centres — closer to real embeddings than uniform noise.
static std::vector<std::vector<float>> makeData(int n, int dims, int clusters, std::mt19937& rng) {
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    std::normal_distribution<float> N(0.0f, 0.15f);
    std::vector<std::vector<float>> centres(clusters, std::vector<float>(dims));
    for (auto& c : centres) for (auto& x : c) x = U(rng);
    std::uniform_int_distribution<int> pick(0, clusters - 1);
    std::vector<std::vector<float>> out(n, std::vector<float>(dims));
    for (auto& v : out) {
        auto& c = centres[pick(rng)];
        for (int d = 0; d < dims; d++) v[d] = c[d] + N(rng);
    }
    return out;
}

struct Row { int dims; double bfMs, kdMs, hnswMs, kdRecall, hnswRecall, kdBuild, hnswBuild; };

static Row runOne(int n, int dims, int nq, int k, int ef) {
    std::mt19937 rng(7);
    // Queries are drawn from the same distribution as the data (held out, not inserted).
    auto data = makeData(n + nq, dims, 50, rng);
    std::vector<std::vector<float>> queries(data.end() - nq, data.end());
    data.resize(n);
    DistFn dist  = euclidean;

    BruteForce bf;
    KDTree kd(dims);
    HNSW hnsw(16, 200);

    for (int i = 0; i < n; i++) bf.insert({i, "", "", data[i]});

    auto t = Clock::now();
    for (int i = 0; i < n; i++) kd.insert({i, "", "", data[i]});
    double kdBuild = msSince(t);

    t = Clock::now();
    for (int i = 0; i < n; i++) hnsw.insert({i, "", "", data[i]}, dist);
    double hnswBuild = msSince(t);

    double bfT = 0, kdT = 0, hnT = 0, kdHit = 0, hnHit = 0;
    for (auto& q : queries) {
        t = Clock::now(); auto truth = bf.knn(q, k, dist);       bfT += msSince(t);
        t = Clock::now(); auto kr    = kd.knn(q, k, dist);       kdT += msSince(t);
        t = Clock::now(); auto hr    = hnsw.knn(q, k, ef, dist); hnT += msSince(t);

        std::set<int> ids;
        for (auto& p : truth) ids.insert(p.second);
        for (auto& p : kr) kdHit += ids.count(p.second);
        for (auto& p : hr) hnHit += ids.count(p.second);
    }
    double total = (double)nq * k;
    return {dims, bfT / nq, kdT / nq, hnT / nq,
            100.0 * kdHit / total, 100.0 * hnHit / total, kdBuild, hnswBuild};
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 10000;
    std::vector<int> dimsList;
    for (int i = 2; i < argc; i++) dimsList.push_back(std::atoi(argv[i]));
    if (dimsList.empty()) dimsList = {16, 128, 768};

    const int nq = 200, k = 10, ef = 50;
    std::printf("N=%d vectors, %d queries, k=%d, HNSW(M=16, efConstruction=200, efSearch=%d), Euclidean\n\n",
                n, nq, k, ef);
    std::printf("| Dims | Brute Force (ms/query) | KD-Tree (ms/query) | HNSW (ms/query) | HNSW speedup vs BF | KD-Tree recall@%d | HNSW recall@%d | HNSW build (s) |\n", k, k);
    std::printf("|---|---|---|---|---|---|---|---|\n");
    for (int d : dimsList) {
        Row r = runOne(n, d, nq, k, ef);
        std::printf("| %d | %.3f | %.3f | %.3f | %.1fx | %.1f%% | %.1f%% | %.1f |\n",
                    r.dims, r.bfMs, r.kdMs, r.hnswMs, r.bfMs / r.hnswMs,
                    r.kdRecall, r.hnswRecall, r.hnswBuild / 1000.0);
        std::fflush(stdout);
    }
    return 0;
}
