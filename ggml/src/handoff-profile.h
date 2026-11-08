// Optional host-side timing. Existing synchronization is unchanged.
#include <string>
struct handoff_record {
    int64_t start, fences, copies, compute, end;
    int64_t source_wait = 0, destination_wait = 0, copy = 0, async_try = 0;
    int64_t prewait = 0;
    int split, nodes, inputs;
    int backend_calls = 1;
    char backend[32], first[GGML_MAX_NAME];
};
static std::vector<handoff_record> handoff_records;
static const char * handoff_path = nullptr;
static bool handoff_initialized = false, handoff_stopped = false;
static int64_t handoff_now() {
#ifdef _WIN32
    LARGE_INTEGER q, f; QueryPerformanceCounter(&q); QueryPerformanceFrequency(&f);
    return (int64_t) (double(q.QuadPart) * 1e6 / f.QuadPart);
#else
    return ggml_time_us();
#endif
}
static void handoff_init() {
    if (handoff_initialized) return;
    handoff_initialized = true;
    handoff_path = getenv("GGML_HANDOFF_PROFILE");
    if (handoff_path) handoff_records.reserve(1048576);
}
static void handoff_flush() {
    if (!handoff_path || handoff_stopped) return;
    const std::string trigger = std::string(handoff_path) + ".flush";
    FILE * request = fopen(trigger.c_str(), "rb");
    if (!request && handoff_records.size() < 1048000) return;
    if (request) fclose(request);
    FILE * f = fopen(handoff_path, "wb");
    if (!f) return;
    fprintf(f, "start_us,fences_us,copies_us,compute_us,end_us,source_wait_us,destination_wait_us,copy_us,async_try_us,prewait_us,split,nodes,inputs,backend_calls,backend,first\n");
    for (const auto & r : handoff_records) {
        fprintf(f, "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d,%d,%d,%d,%s,%s\n",
            (long long)r.start, (long long)r.fences, (long long)r.copies, (long long)r.compute, (long long)r.end,
            (long long)r.source_wait, (long long)r.destination_wait, (long long)r.copy, (long long)r.async_try,
            (long long)r.prewait, r.split, r.nodes, r.inputs, r.backend_calls, r.backend, r.first);
    }
    fclose(f); handoff_stopped = true;
}
