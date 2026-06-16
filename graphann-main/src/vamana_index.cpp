#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <limits>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search (With Prefetching & Warm-Start)
// ============================================================================

// ============================================================================
// Greedy Search (Adaptive Beam Length + Prefetching + Warm-Start)
// ============================================================================

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L_initial, 
                           const std::vector<uint32_t>& init_nodes, 
                           uint32_t L_max) const {
    
    // If L_max wasn't provided, default it to L_initial (for standard build phase)
    if (L_max == 0 || L_max < L_initial) L_max = L_initial;

    std::set<Candidate> candidate_set;
    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;
    std::set<uint32_t> expanded;

    // Warm-Start Initialization
    if (init_nodes.empty()) {
        float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
        dist_cmps++;
        candidate_set.insert({start_dist, start_node_});
        visited[start_node_] = true;
    } else {
        for (uint32_t id : init_nodes) {
            if (!visited[id]) {
                float d = compute_l2sq(query, get_vector(id), dim_);
                dist_cmps++;
                candidate_set.insert({d, id});
                visited[id] = true;
            }
        }
    }

    uint32_t current_expansion_limit = L_initial;
    uint32_t expansion_count = 0;

    while (true) {
        // 1. Trap Detection: Have we hit our current expansion limit?
        if (expansion_count >= current_expansion_limit) {
            // We hit the limit. But are we stuck?
            // If we haven't reached the absolute L_max, we trigger the escape hatch!
            if (current_expansion_limit < L_max) {
                current_expansion_limit = L_max; // Widen the beam!
            } else {
                break; // We hit the hard L_max limit. Stop searching.
            }
        }

        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                best_node = id;
                break;
            }
        }
        
        // If no more unvisited nodes exist in our pocket, we are done.
        if (best_node == UINT32_MAX) break;

        expanded.insert(best_node);
        expansion_count++;

        std::vector<uint32_t> neighbors;
        {
            locks_[best_node].lock();
            neighbors = graph_[best_node];
            locks_[best_node].unlock();
        }

        for (size_t i = 0; i < neighbors.size(); i++) {
            // Memory Prefetching
            if (i + 2 < neighbors.size()) {
                __builtin_prefetch(get_vector(neighbors[i + 2]), 0, 1);
            }

            uint32_t nbr = neighbors[i];
            if (visited[nbr]) continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            // Notice we use L_max here, so we don't accidentally throw away 
            // nodes that we might need if the trap is triggered later!
            if (candidate_set.size() < L_max) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
        }
    }

    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}
// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================

std::vector<uint32_t> VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                                                float alpha, uint32_t R) {
    // Remove self from candidates if present
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Sort by distance to node (ascending)
    std::sort(candidates.begin(), candidates.end());

    // Deduplicate by node ID (crucial for Pass 2 where candidates may merge with existing edges)
    candidates.erase(
        std::unique(candidates.begin(), candidates.end(),
                    [](const Candidate& a, const Candidate& b) {
                        return a.second == b.second;
                    }),
        candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R)
            break;

        // Check alpha-RNG condition against all already-selected neighbors
        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }

        if (keep)
            new_neighbors.push_back(cand_id);
    }

    // Return the edges instead of mutating graph_[node] directly
    return new_neighbors;
}

// ============================================================================
// Build (Pro 2-Pass Implementation)
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    // --- Find the Medoid (Start Node) ---
    std::cout << "Calculating geometric medoid for start node..." << std::endl;
    
    // 1. Calculate the mean vector of a random sample
    std::vector<float> centroid(dim_, 0.0f);
    uint32_t sample_size = std::min(npts_, 10000u);
    for (uint32_t i = 0; i < sample_size; i++) {
        const float* vec = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++) {
            centroid[d] += vec[d];
        }
    }
    for (uint32_t d = 0; d < dim_; d++) {
        centroid[d] /= sample_size;
    }

    // 2. Find the actual data point closest to the centroid
    float min_dist = std::numeric_limits<float>::max();
    start_node_ = 0;
    for (uint32_t i = 0; i < npts_; i++) {
        float dist = compute_l2sq(centroid.data(), get_vector(i), dim_);
        if (dist < min_dist) {
            min_dist = dist;
            start_node_ = i;
        }
    }
    std::cout << "  Start node set to true medoid: " << start_node_ << std::endl;

    if (L < R) L = R;

    // Pre-allocate graph and initialize array of SpinLocks
    graph_.resize(npts_);
    locks_ = std::make_unique<SpinLock[]>(npts_);

    // Pre-allocate edge capacity to prevent memory reallocations during lock
    for(size_t i = 0; i < npts_; i++) {
        graph_[i].reserve(static_cast<uint32_t>(gamma * R) + 1);
    }

    std::mt19937 rng(42); // RNG required for shuffling insertion order

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    Timer build_timer;

    for (int pass = 1; pass <= 2; pass++) {
        std::cout << "\nStarting Pass " << pass << " of 2..." << std::endl;
        float current_alpha = (pass == 1) ? 1.0f : alpha;
        
        std::vector<uint32_t> perm(npts_);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t idx = 0; idx < npts_; idx++) {
            uint32_t point = perm[idx];

            std::vector<uint32_t> warm_start_nodes;
            if (pass == 2) {
                // Pass 2: Extract existing neighbors to warm-start the search
                locks_[point].lock();
                warm_start_nodes = graph_[point];
                locks_[point].unlock();
            }

            // Execute greedy search (Uses warm-start if pass 2)
            auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L, warm_start_nodes);

            // In Pass 2, ensure current neighbors are guaranteed consideration
            if (pass == 2) {
                for (uint32_t nbr : warm_start_nodes) {
                    float d = compute_l2sq(get_vector(point), get_vector(nbr), dim_);
                    candidates.push_back({d, nbr});
                }
            }

            std::vector<uint32_t> new_edges = robust_prune(point, candidates, current_alpha, R);

            locks_[point].lock();
            graph_[point] = new_edges;
            locks_[point].unlock();

            for (uint32_t nbr : new_edges) {
                locks_[nbr].lock();
                if (std::find(graph_[nbr].begin(), graph_[nbr].end(), point) == graph_[nbr].end()) {
                    graph_[nbr].push_back(point);
                }
                
                if (graph_[nbr].size() > gamma_R) {
                    std::vector<Candidate> nbr_candidates;
                    nbr_candidates.reserve(graph_[nbr].size());
                    for (uint32_t nn : graph_[nbr]) {
                        float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                        nbr_candidates.push_back({d, nn});
                    }
                    
                    // Note: Pruning under lock is safe here because we own nbr's lock
                    graph_[nbr] = robust_prune(nbr, nbr_candidates, current_alpha, R);
                }
                locks_[nbr].unlock();
            }

            if (idx % 10000 == 0) {
                #pragma omp critical
                {
                    std::cout << "\r  Pass " << pass << " - Inserted " << idx << " / " << npts_
                              << " points" << std::flush;
                }
            }
        }
        std::cout << "\r  Pass " << pass << " - Inserted " << npts_ << " / " << npts_ << " points" << std::endl;
    }

    double build_time = build_timer.elapsed_seconds();
    
    // Compute average degree
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();
    double avg_degree = (double)total_edges / npts_;

    std::cout << "\n  Build complete in " << build_time << " seconds." << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;
}

// ============================================================================
// Search
// ============================================================================

// ============================================================================
// Search (Adaptive Beam Length)
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L_initial, uint32_t L_max) const {
    // Safety checks: Ensure beams are at least as large as K, and L_max >= L_initial
    if (L_initial < K) L_initial = K;
    if (L_max < L_initial) L_max = L_initial;
    
    Timer t;
    // Pass the adaptive parameters straight into the engine
    auto [candidates, dist_cmps] = greedy_search(query, L_initial, {}, L_max);
    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Save / Load
// ============================================================================

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0) {
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts), 4);
    in.read(reinterpret_cast<char*>(&file_dim), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::make_unique<SpinLock[]>(npts_);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0) {
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}