/********************************************************
 * Author: Namcheol Lee
 * Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * Contact: nclee@redwood.snu.ac.kr
 * Date: 2025-07-02
 * Description: Priority Kernel Queue Implementation
 ********************************************************/

// Priority kernel queue is a lock-free single-producer, single-consumer queue
template<typename T, size_t N = 2048>
struct PriorityKernelQueue {
    static_assert((N & (N - 1)) == 0, "N must be power-of-two");
    std::atomic<size_t> head{0};           // written by consumer
    std::atomic<size_t> tail{0};           // written by producer
    T buf[N];

    bool push(T&& v, bool& was_empty) {
        size_t h = head.load(std::memory_order_acquire);
        size_t t = tail.load(std::memory_order_relaxed);
        if (t - h == N) return false;      // full

        was_empty = (t == h);
        buf[t & (N - 1)] = std::move(v);
        tail.store(t + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& out) {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_acquire);
        if (h == t) return false;          // empty

        out = std::move(buf[h & (N - 1)]);
        head.store(h + 1, std::memory_order_release);
        return true;
    }

    bool peek(T& out) const {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_acquire);
        if (h == t) return false;          // empty

        out = buf[h & (N - 1)];
        return true;
    }

    bool commit_pop() {
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_acquire);
        if (h == t) return false;          // empty

        head.store(h + 1, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head.load(std::memory_order_relaxed) ==
               tail.load(std::memory_order_relaxed);
    }
};