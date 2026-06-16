#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <atomic>

// Result of a single query search.
struct SearchResult {
    std::vector<uint32_t> ids;  // nearest neighbor IDs (sorted by distance)
    uint32_t dist_cmps;         // number of distance computations
    double latency_us;          // search latency in microseconds
};

// Ultra-lightweight SpinLock to replace heavy std::mutex
struct SpinLock {
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
    void lock() {
        while (locked.test_and_set(std::memory_order_acquire)) {
            // CPU pause hint (prevents pipeline flush and reduces power)
#if defined(__i386__) || defined(__x86_64__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
#endif
        }
    }
    void unlock() {
        locked.clear(std::memory_order_release);
    }
};

class VamanaIndex {
  public:
    VamanaIndex() = default;
    ~VamanaIndex();

    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, float gamma);

    SearchResult search(const float* query, uint32_t K, uint32_t L_initial, uint32_t L_max) const;

    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_; }

  private:
    // --------------------------------------------------------
    // REPLACED the old std::pair with this custom struct. 
    // This allows std::set to handle custom sorting safely.
    // --------------------------------------------------------
    struct Candidate {
        float first;     // distance
        uint32_t second; // id
        bool operator<(const Candidate& other) const {
            if (first != other.first) return first < other.first;
            return second < other.second;
        }
    };

    // --------------------------------------------------------
    // SINGLE declaration of greedy_search with adaptive L logic
    // --------------------------------------------------------
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L_initial, 
                  const std::vector<uint32_t>& init_nodes = {},
                  uint32_t L_max = 0) const;

    std::vector<uint32_t> robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                                       float alpha, uint32_t R);

    float* data_      = nullptr;
    uint32_t npts_    = 0;
    uint32_t dim_     = 0;
    bool   owns_data_ = false;

    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    // High-performance SpinLock array
    mutable std::unique_ptr<SpinLock[]> locks_;

    const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};