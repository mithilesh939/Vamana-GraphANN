#include "vamana_index.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index <index_path>\n"
              << " --data <fbin_path>\n"
              << " --queries <query_fbin_path>\n"
              << " --gt <ground_truth_ibin_path>\n"
              << " [--K <num_neighbors=10>]\n"
              << " [--L_initial <fast_beam=20>]\n"
              << " [--L_max <safety_beam=150>]\n"
              << std::endl;
}

// Compute recall@K: fraction of true top-K neighbors found in result
static double compute_recall(const std::vector<uint32_t>& result,
                             const uint32_t* gt, uint32_t K) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < K && i < result.size(); i++) {
        for (uint32_t j = 0; j < K; j++) {
            if (result[i] == gt[j]) {
                found++;
                break;
            }
        }
    }
    return (double)found / K;
}

int main(int argc, char** argv) {
    std::string index_path, data_path, query_path, gt_path;
    uint32_t K = 10;
    uint32_t L_initial = 20; // Default fast beam
    uint32_t L_max = 150;    // Default safety net beam

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc)         index_path = argv[++i];
        else if (arg == "--data" && i + 1 < argc)      data_path = argv[++i];
        else if (arg == "--queries" && i + 1 < argc)   query_path = argv[++i];
        else if (arg == "--gt" && i + 1 < argc)        gt_path = argv[++i];
        else if (arg == "--K" && i + 1 < argc)         K = std::atoi(argv[++i]);
        else if (arg == "--L_initial" && i + 1 < argc) L_initial = std::atoi(argv[++i]);
        else if (arg == "--L_max" && i + 1 < argc)     L_max = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Only check for required file paths
    if (index_path.empty() || data_path.empty() || query_path.empty() || gt_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // --- Load index ---
    std::cout << "Loading index..." << std::endl;
    VamanaIndex index;
    index.load(index_path, data_path);

    // --- Load queries ---
    std::cout << "Loading queries from " << query_path << "..." << std::endl;
    FloatMatrix queries = load_fbin(query_path);
    std::cout << "  Queries: " << queries.npts << " x " << queries.dims << std::endl;

    if (queries.dims != index.get_dim()) {
        std::cerr << "Error: query dimension (" << queries.dims
                  << ") != index dimension (" << index.get_dim() << ")" << std::endl;
        return 1;
    }

    // --- Load ground truth ---
    std::cout << "Loading ground truth from " << gt_path << "..." << std::endl;
    IntMatrix gt = load_ibin(gt_path);
    std::cout << "  Ground truth: " << gt.npts << " x " << gt.dims << std::endl;

    if (gt.npts != queries.npts) {
        std::cerr << "Error: ground truth rows (" << gt.npts
                  << ") != number of queries (" << queries.npts << ")" << std::endl;
        return 1;
    }
    if (gt.dims < K) {
        std::cerr << "Warning: ground truth has " << gt.dims
                  << " neighbors per query but K=" << K << std::endl;
        K = gt.dims;
    }

    uint32_t nq = queries.npts;

    // --- Run Adaptive Search ---
    std::cout << "\n=== Adaptive Search Results (K=" << K << ") ===" << std::endl;
    std::cout << "L_initial: " << L_initial << " | L_max: " << L_max << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    std::vector<double> recalls(nq);
    std::vector<uint32_t> dist_cmps(nq);
    std::vector<double> latencies(nq);

    #pragma omp parallel for schedule(dynamic, 16)
    for (uint32_t q = 0; q < nq; q++) {
        SearchResult res = index.search(queries.row(q), K, L_initial, L_max);

        recalls[q] = compute_recall(res.ids, gt.row(q), K);
        dist_cmps[q] = res.dist_cmps;
        latencies[q] = res.latency_us;
    }

    // Aggregate statistics
    double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / nq;
    double avg_cmps = (double)std::accumulate(dist_cmps.begin(), dist_cmps.end(), 0ULL) / nq;
    double avg_lat = std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;

    // P99 latency
    std::sort(latencies.begin(), latencies.end());
    double p99_lat = latencies[(size_t)(0.99 * nq)];

    std::cout << "Recall@" << K << "      : " << std::fixed << std::setprecision(4) << avg_recall << "\n"
              << "Avg Dist Cmps  : " << std::fixed << std::setprecision(1) << avg_cmps << "\n"
              << "Avg Latency    : " << std::fixed << std::setprecision(1) << avg_lat << " us\n"
              << "P99 Latency    : " << std::fixed << std::setprecision(1) << p99_lat << " us\n";
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Done." << std::endl;

    return 0;
}