#pragma once

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>

#define MIN(A,B) (((A)>(B))?(B):(A))
#define MAX(A,B) (((A)<(B))?(B):(A))
#define ABS(A) (((A)<0)?(-(A)):(A))

const uint64_t FACT[21] = {
    1,
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    479001600,
    6227020800,
    87178291200,
    1307674368000,
    20922789888000,
    355687428096000,
    6402373705728000,
    121645100408832000,
    2432902008176640000,
};

typedef struct {
    char title[1024];
    char description[1024];
    char seq[128];
    float modseq[128];
    uint32_t length;
    float calc_mass;
    float mass;
    float rt;
    uint32_t charge;
    double score;
    char score_data[10240];
} res;

typedef struct {
    char description[1024];
    char sequence[128];
    float mods[128];
    float rt;
    uint32_t length;
    uint32_t npeaks;
    float mass;
    char meta[10240];
    float spec[][2];
} db;

typedef struct {
    char desc[1024];
    char seq[128];
    float mods[128];
} predb;

typedef struct {
    char title[1024];
    float rt;
    uint32_t charge;
    float mass;
    uint32_t npeaks;
    float min_mass;
    float max_mass;
    char meta[10240];
    float spec[][2];
} query;

typedef struct {
    uint64_t n;
    uint64_t start;
    void* data;
} ret;

typedef struct {
    int64_t start;
    int64_t end;
} range;

typedef struct {
    char* fname1;
    char* fname2;
    char* fname_out;
    uint64_t size;
    uint64_t batch_size;
} merge_payload;

typedef struct {
    char* fname;
    uint64_t start;
    uint64_t size;
} sort_payload;

typedef enum {
    JOB_NONE = 0,
    JOB_SORT,
    JOB_MERGE
} job_type;

typedef struct {
    void* payload;
    job_type type;
    char done;
    void* ret;
} task;

typedef struct _task_node {
    task* task;
    struct _task_node* next;
    struct _task_node* prev;
} task_node;

typedef struct {
    pthread_t* threads;
    uint32_t n_threads;
    task_node work_queue;
    task_node* work_tail;
    pthread_mutex_t queue_lock;
    char die;
} thread_pool;

typedef struct {
    double* distances;
    char* mask;
    double* scores;
    double* sumI;
    uint32_t* total_matched;
    float* theoretical;
    float* spec;
    uint32_t ncands;
    uint32_t npeaks;
} score_ret;

uint64_t dump(char*, uint64_t, void*, uint64_t, char);
ret load(char*, uint64_t, uint64_t, uint64_t, uint64_t);

uint64_t merge_sort(uint32_t, char**, char*, uint32_t, uint32_t, uint32_t, uint32_t);

range find_data(char*, uint64_t, uint64_t, float, float);

void reorder(char*, char*, char*, uint64_t, uint64_t);

res* make_res(double*, score_ret, char*, float, float, int, db*, int);

typedef struct {
    void* cands;
    int n_cands;
    int npeaks;
    uint64_t elt_size;
    float tol;
    void* q;
} score_data;

void free_ret(ret);
void free_ptr(void*);
void free_score(score_ret*);

thread_pool pool;
//score_ret rnhs(void*, query*, void*, int, int, uint64_t, float);
score_ret* rnhs(score_data);
void* alloc(uint64_t);
char* score_str(score_ret*, int);
