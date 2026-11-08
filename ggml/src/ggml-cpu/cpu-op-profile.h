/* Optional single-context CPU operation timing. No added worker barriers. */
struct diag_cpu_record {
    int64_t start_us, compute_us, end_us;
    int64_t ne0, ne1, ne2, k, m, experts, query_rows;
    uint64_t logical_bytes;
    int op, type, nth, fused, routes, misses;
    char name[GGML_MAX_NAME];
};

static struct diag_cpu_record * diag_cpu_records;
static size_t diag_cpu_count;
static const size_t diag_cpu_capacity = 1048576;
static const char * diag_cpu_path;
static int diag_cpu_initialized, diag_cpu_stopped;
#ifdef _WIN32
static int64_t diag_cpu_frequency;
#endif

static int64_t diag_cpu_now(void) {
#ifdef _WIN32
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (int64_t) ((double) now.QuadPart * 1000000.0 / (double) diag_cpu_frequency);
#else
    return ggml_time_us();
#endif
}

static void diag_cpu_init(void) {
    if (diag_cpu_initialized) return;
    diag_cpu_initialized = 1;
    diag_cpu_path = getenv("GGML_CPU_OP_PROFILE");
    if (!diag_cpu_path || !diag_cpu_path[0]) return;
#ifdef _WIN32
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    diag_cpu_frequency = freq.QuadPart;
#endif
    diag_cpu_records = calloc(diag_cpu_capacity, sizeof(struct diag_cpu_record));
}

static struct diag_cpu_record * diag_cpu_begin(const struct ggml_tensor * node, int nth) {
    if (!diag_cpu_records || diag_cpu_stopped || diag_cpu_count == diag_cpu_capacity) return NULL;
    struct diag_cpu_record * rec = &diag_cpu_records[diag_cpu_count++];
    rec->op = node->op;
    rec->nth = nth;
    rec->ne0 = node->ne[0]; rec->ne1 = node->ne[1]; rec->ne2 = node->ne[2];
    snprintf(rec->name, sizeof(rec->name), "%s", node->name);
    rec->type = node->src[0] ? node->src[0]->type : node->type;
    rec->start_us = diag_cpu_now();
    return rec;
}

static void diag_cpu_end(struct diag_cpu_record * rec, const struct ggml_tensor * node) {
    if (!rec) return;
    rec->end_us = diag_cpu_now();
    if (node->op != GGML_OP_MUL_MAT_ID && node->op != GGML_OP_MUL_MAT_ID_MASKED) return;
    const struct ggml_tensor * weights = node->src[0];
    const struct ggml_tensor * ids = node->src[2];
    const struct ggml_tensor * mask = node->op == GGML_OP_MUL_MAT_ID_MASKED ? node->src[3] : NULL;
    rec->k = weights->ne[0]; rec->m = weights->ne[1]; rec->experts = weights->ne[2];
    rec->query_rows = ids->ne[1];
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        for (int64_t route = 0; route < ids->ne[0]; ++route) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + row*ids->nb[1] + route*ids->nb[0]);
            rec->routes++;
            if (!mask || ((const int32_t *) mask->data)[expert] == 0) rec->misses++;
        }
    }
    rec->logical_bytes = (uint64_t) rec->misses * ggml_row_size(weights->type, weights->ne[0]) * weights->ne[1];
}

static void diag_cpu_maybe_flush(void) {
    if (!diag_cpu_records || diag_cpu_stopped) return;
    static unsigned graph_count;
    if ((++graph_count % 128) != 0 && diag_cpu_count != diag_cpu_capacity) return;
    char trigger[2048];
    snprintf(trigger, sizeof(trigger), "%s.flush", diag_cpu_path);
    FILE * request = fopen(trigger, "rb");
    if (!request && diag_cpu_count != diag_cpu_capacity) return;
    if (request) fclose(request);
    FILE * output = fopen(diag_cpu_path, "wb");
    if (!output) return;
    fprintf(output, "start_us,compute_us,end_us,op,type,threads,fused,name,ne0,ne1,ne2,k,m,experts,query_rows,routes,misses,logical_bytes\n");
    for (size_t i = 0; i < diag_cpu_count; ++i) {
        const struct diag_cpu_record * r = &diag_cpu_records[i];
        fprintf(output, "%lld,%lld,%lld,%s,%s,%d,%d,%s,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%d,%d,%llu\n",
            (long long)r->start_us, (long long)r->compute_us, (long long)r->end_us,
            ggml_op_name(r->op), ggml_type_name(r->type), r->nth, r->fused, r->name,
            (long long)r->ne0, (long long)r->ne1, (long long)r->ne2,
            (long long)r->k, (long long)r->m, (long long)r->experts, (long long)r->query_rows,
            r->routes, r->misses, (unsigned long long)r->logical_bytes);
    }
    fclose(output);
    diag_cpu_stopped = 1;
}
