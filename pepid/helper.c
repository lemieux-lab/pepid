#include "helper.h"
/*
// Given key data and an array of indices, reorder the elements of data so they fit the indices
void reorder_key(float* data, uint64_t* indices, int n, uint64_t offset) {
    for(int i = 0; i < n; i++) {
        uint64_t ii = i;
        uint64_t jj = indices[i] - offset;

        float tmp = data[ii];
        while(ii != jj) {
            data[ii] = data[jj];
            indices[ii] = ii + offset;
            ii = jj;
            jj = indices[ii] - offset;
        }
        data[ii] = tmp;
        indices[ii] = ii + offset;
    }
}

// reorders the seq file entries in fname_tgt as per the indices in fname_idx
void reorder(char* fname_idx, char* fname_tgt, char* out, uint64_t out_size, uint64_t batch_size) {
    db* buff = calloc(out_size, batch_size);
    uint64_t* idx = NULL;
    predb* val = NULL;

    FILE* f = fopen(fname_idx, "rb");
    fseeko(f, 0, SEEK_END);
    uint64_t idx_size = ftello(f);
    uint64_t fsize = idx_size / sizeof(uint64_t); 
    fclose(f);

    uint64_t end_idx = batch_size;

    for(uint64_t i = 0; (i < fsize) && (i % batch_size < end_idx); i++) {
        if(i % batch_size == 0) {
            if(i != 0) {
                for(uint64_t j = 0; j < batch_size; j++) {
                    dump(out, idx[j] * out_size, &(buff[j]), out_size, 0);
                }
            }
            ret ri = load(fname_idx, i * sizeof(uint64_t), sizeof(uint64_t), batch_size, 0);
            ret rv = load(fname_tgt, i * sizeof(predb), sizeof(predb), batch_size, 0);
            end_idx = ri.n / sizeof(uint64_t);

            free(idx); free(val);
            idx = (uint64_t*)(ri.data);
            val = (predb*)(rv.data);
        }

        if(i % batch_size >= end_idx) {
            for(uint64_t j = 0; j < end_idx; j++) {
                dump(out, idx[j] * out_size, &(buff[j]), out_size, 0);
            }
            break;
        } else {
            memcpy(buff[i % batch_size].description, val[i % batch_size].desc, 1024*sizeof(char));
            memcpy(buff[i % batch_size].sequence, val[i % batch_size].seq, 128*sizeof(char));
            memcpy(buff[i % batch_size].mods, val[i % batch_size].mods, 128*sizeof(float));
        }
    }

    free(buff); free(val); free(idx);
}

uint64_t dump(char* fname, uint64_t file_offset, void* data, uint64_t total_size, char erase) {
    FILE* outf = fopen(fname, erase? "wb" : "r+b");
    fseeko(outf, file_offset, SEEK_SET);
    uint64_t written = fwrite(data, 1, total_size, outf);
    fclose(outf);
    return written;
}

ret load(char* fname, uint64_t offset, uint64_t elt_size, uint64_t n_elts, uint64_t start) {
    FILE* inf = fopen(fname, "rb");

    struct stat s;
    fstat(fileno(inf), &s);

    uint64_t size = elt_size * n_elts;
    uint64_t size_to_read = (size == 0? s.st_size - offset:MIN(size,s.st_size - offset));

    fseeko(inf, offset, SEEK_SET);
    void* out = calloc(1, size_to_read);
    uint64_t actually_read = fread(out, 1, size_to_read, inf);

    fclose(inf);

    return (ret){ .n = actually_read, .data = out, .start = (offset / elt_size + start) };
}

char* make_fname(char* src, char idx) {
    int lgt = strlen(src);
    char* ret = calloc(lgt + 1 + 4 + (idx? 4 : 0), 1);
    if(!idx) {
        sprintf(ret, "%s.bin", src);
    } else {
        sprintf(ret, "%s_idx.bin", src);
    }
    return ret;
}

void merge_sort_merge(char* fname1, char* fname2, char* out, uint32_t size, uint32_t batch_size) {
    char* f1 = make_fname(fname1, 0);
    char* i1 = make_fname(fname1, 1);
    char* f2 = make_fname(fname2, 0);
    char* i2 = make_fname(fname2, 1);
    char* fo = make_fname(out, 0);
    char* io = make_fname(out, 1);

    void* buff_out = calloc(size, batch_size);
    void* buff1 = NULL;
    void* buff2 = NULL;

    void* buff_out_idx = calloc(sizeof(uint64_t), batch_size);
    void* buff1_idx = NULL;
    void* buff2_idx = NULL;

    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t k = 0;

    FILE* data1 = fopen(f1, "rb");
    FILE* data2 = fopen(f2, "rb");

    {
    FILE* fo_file = fopen(fo, "wb");
    FILE* io_file = fopen(io, "wb");
    fclose(io_file);
    fclose(fo_file);
    }

    fseeko(data1, 0, SEEK_END);
    uint64_t f1_size = ftello(data1) / size;
    fclose(data1);

    fseeko(data2, 0, SEEK_END);
    uint64_t f2_size = ftello(data2) / size;
    fclose(data2);

    uint64_t f1_start = 0;
    uint64_t f2_start = 0;
    uint64_t fo_start = 0;

    while((i < f1_size) || (j < f2_size)) {
        if((buff1 == NULL) || ((i - f1_start) >= batch_size)) {
            ret r = load(f1, i * size, size, batch_size, 0);
            ret r_idx = load(i1, i * sizeof(uint64_t), sizeof(uint64_t), batch_size, 0);
            free(buff1); free(buff1_idx);
            buff1 = r.data;
            buff1_idx = r_idx.data;
            f1_start = i;
        }
        if((buff2 == NULL) || ((j - f2_start) >= batch_size)) {
            ret r = load(f2, j * size, size, batch_size, 0);
            ret r_idx = load(i2, j * sizeof(uint64_t), sizeof(uint64_t), batch_size, 0);
            free(buff2); free(buff2_idx);
            buff2 = r.data;
            buff2_idx = r_idx.data;
            f2_start = j;
        }
        while((i - f1_start < batch_size) && (j - f2_start < batch_size) && (i < f1_size || j < f2_size)) {
            if(i >= f1_size) {
                float v2 = ((float*)buff2)[j - f2_start];
                uint64_t u2 = ((uint64_t*)buff2_idx)[j - f2_start];
                ((float*)buff_out)[k - fo_start] = v2;
                ((uint64_t*)buff_out_idx)[k - fo_start] = u2;
                j++;
            } else if(j >= f2_size) {
                float v1 = ((float*)buff1)[i - f1_start];
                uint64_t u1 = ((uint64_t*)buff1_idx)[i - f1_start];
                ((float*)buff_out)[k - fo_start] = v1;
                ((uint64_t*)buff_out_idx)[k - fo_start] = u1;
                i++;
            } else {
                float v1 = ((float*)buff1)[i - f1_start];
                float v2 = ((float*)buff2)[j - f2_start];
                uint64_t u1 = ((uint64_t*)buff1_idx)[i - f1_start];
                uint64_t u2 = ((uint64_t*)buff2_idx)[j - f2_start];
                
                if(v1 <= v2) {
                    ((float*)buff_out)[k - fo_start] = v1;
                    ((uint64_t*)buff_out_idx)[k - fo_start] = u1;
                    i++;
                } else {
                    ((float*)buff_out)[k - fo_start] = v2;
                    ((uint64_t*)buff_out_idx)[k - fo_start] = u2;
                    j++;
                }
            }

            k++;
            if((k - fo_start) >= batch_size) {
                dump(fo, fo_start * size, buff_out, (k - fo_start) * size,  0);
                dump(io, fo_start * sizeof(uint64_t), buff_out_idx, (k - fo_start) * sizeof(uint64_t),  0);
                fo_start = k;
            }
        }
    }

    if(k > fo_start) {
        dump(fo, fo_start * size, buff_out, (k - fo_start) * size,  0);
        dump(io, fo_start * sizeof(uint64_t), buff_out_idx, (k - fo_start) * sizeof(uint64_t),  0);
    }

    free(buff1); free(buff1_idx);
    free(buff2); free(buff2_idx);
    free(buff_out); free(buff_out_idx);

    free(f1); free(i1);
    free(f2); free(i2);
    free(fo); free(io);
}

int float_cmp(const void* left, const void* right) {
    float l = *((float*)left);
    float r = *((float*)right);
    return (r < l? 1 : (r == l? 0 : -1));
}

int merge_sort_cmp_db(const void* left, const void* right) {
    db* rleft = (db*)left;
    db* rright = (db*)right;
    float a = rright->mass;
    float b = rleft->mass;
    return (a < b? 1 : (a == b? 0 : -1));
}

int merge_argsort_cmp_db(const void* left, const void* right, void* arg) {
    ret* rarg = (ret*)arg;
    float* arr = (float*)(rarg->data);
    uint64_t* rleft = (uint64_t*)left;
    uint64_t* rright = (uint64_t*)right;
    float a = arr[(*rright) - rarg->start];
    float b = arr[(*rleft) - rarg->start];
    return (a < b? 1 : (a == b? 0 : -1));
}

uint64_t merge_sort_sort(char* fname, uint32_t size, uint64_t start) {
    char* full_name = make_fname(fname, 0);
    char* idx_name = make_fname(fname, 1);

    ret r = load(full_name, 0, size, 0, start);

    uint64_t base = r.start;
    uint64_t* idx_arr = calloc(r.n / size, sizeof(uint64_t));
    for(int i = 0; i < r.n / size; i++) {
        idx_arr[i] = base + i;
    }

    qsort_r(idx_arr, r.n / size, sizeof(uint64_t), &merge_argsort_cmp_db, (void*)(&r));
    reorder_key((float*)(r.data), idx_arr, r.n / size, base);

    uint64_t dump_size = dump(full_name, 0, r.data, r.n, 1);
    dump(idx_name, 0, idx_arr, (r.n / size) * sizeof(uint64_t), 1);

    free(r.data);
    free(idx_arr);
    free(idx_name);
    free(full_name);

    return dump_size;
}

void push_job(task* task) {
    if(!pthread_mutex_lock(&(pool.queue_lock))) {
        pool.work_tail->next = calloc(sizeof(task_node), 1);
        pool.work_tail->next->task = task;
        pool.work_tail->next->prev = pool.work_tail;
        pool.work_tail = pool.work_tail->next;
        pthread_mutex_unlock(&(pool.queue_lock));
    } else {
        printf("[WARN] Failed to acquire lock on queue, giving up.\n");
    }
}

task* try_pop_job() {
    task* ret_task = NULL;
    if(!pthread_mutex_trylock(&(pool.queue_lock))) {
        task_node* ret = pool.work_queue.next;
        if(ret != NULL) {
            pool.work_queue.next = ret->next;
            if(ret->next != NULL) {
                ret->next->prev = &(pool.work_queue);
            } else {
                pool.work_tail = &(pool.work_queue);
            }
            ret_task = ret->task;
            free(ret);
        }
        pthread_mutex_unlock(&(pool.queue_lock));
    }
    return ret_task;
}

void* pool_task(void* arg) {
    while(!pool.die) {
        task* next_task = try_pop_job();
        if(next_task != NULL) { 
            switch(next_task->type) {
                case JOB_SORT: {
                    sort_payload* payload = (sort_payload*)(next_task->payload);
                    long ret = merge_sort_sort(payload->fname, payload->size, payload->start);
                    next_task->ret = (void*)ret; // hack!
                    next_task->done = 1;
                    break;
                }
                case JOB_MERGE: {
                    merge_payload* payload = (merge_payload*)(next_task->payload);
                    merge_sort_merge(payload->fname1, payload->fname2, payload->fname_out, payload->size, payload->batch_size);
                    char* full_fname1 = make_fname(payload->fname1, 0);
                    char* full_fname2 = make_fname(payload->fname2, 0);
                    char* full_idx1 = make_fname(payload->fname1, 1);
                    char* full_idx2 = make_fname(payload->fname2, 1);
                    int status1 = remove(full_fname1); int status2 = remove(full_fname2); int status3 = remove(full_idx1); int status4 = remove(full_idx2);
                    if(!status1 && !status2 && !status3 && !status4) {
                        next_task->done = 1;
                    } else {
                        printf("Error while deleting %s and %s!\n", payload->fname1, payload->fname2);
                    }
                    free(full_fname1); free(full_fname2); free(full_idx1); free(full_idx2);
                    break;
                }
            }
        } else {
            sleep(1); 
        }
    }
    return NULL;
}

uint64_t count(char* fname, uint64_t size) {
    FILE* inf = fopen(fname, "rb");

    struct stat s;
    fstat(fileno(inf), &s);

    return s.st_size / size;
}

uint64_t merge_sort(uint32_t nfiles, char** fnames, char* out, uint32_t size, uint32_t n_threads, uint32_t batch_size, uint32_t n_merge_threads) {
    pool.threads = calloc(sizeof(pthread_t), n_threads);
    pool.n_threads = n_threads;
    pool.work_tail = &(pool.work_queue);
    pthread_mutex_init(&(pool.queue_lock), NULL);

    for(int i = 0; i < n_threads; i++) {
        pthread_create(&(pool.threads[i]), NULL, pool_task, NULL);
    }

    uint32_t files_left = nfiles;
    char** fnames_left = calloc(sizeof(char*), files_left);
    uint64_t total_out_size = 0;
    task_node head_task;
    head_task.next = NULL;
    head_task.prev = NULL;
    head_task.task = calloc(sizeof(task), 1);
    head_task.task->done = 1;

    task_node* this_task = &head_task;

    uint64_t start = 0;
    for(int i = 0; i < nfiles; i++) {
        int lgt = strlen(fnames[i]);
        fnames_left[i] = calloc(sizeof(char), lgt+1);
        strncpy(fnames_left[i], fnames[i], lgt);
        this_task->next = calloc(sizeof(task_node), 1);
        this_task->next->prev = this_task;
        this_task = this_task->next;
        this_task->task = calloc(sizeof(task), 1);
        this_task->task->type = JOB_SORT;
        this_task->task->payload = calloc(sizeof(sort_payload), 1);

        ((sort_payload*)(this_task->task->payload))->fname = fnames[i];
        ((sort_payload*)(this_task->task->payload))->start = start;
        ((sort_payload*)(this_task->task->payload))->size = size;

        char* this_fname = make_fname(fnames[i], 0);
        FILE* f = fopen(this_fname, "rb");
        fseeko(f, 0, SEEK_END);
        start += ftello(f) / size;
        fclose(f);
        free(this_fname);

        push_job(this_task->task);
    }

    for(;;) {
        this_task = &head_task;
        char done = 1;
        while(this_task != NULL) {
            done = done && this_task->task->done;
            if(this_task->task->done && this_task->task->payload != NULL) {
                task_node* next_task = this_task->next;
                total_out_size += (uint64_t)this_task->task->ret;       
                free(this_task->task->payload);
                this_task->task->payload = NULL;
                this_task = next_task;
            } else { this_task = this_task->next; }
        }
        if(done) { break; }
        sleep(0.1);
    }

    pool.die = 1;
    for(int i = 0; i < pool.n_threads; i++) {
        pthread_join(pool.threads[i], NULL);
    }
    free(pool.threads);

    while(pool.work_tail != &(pool.work_queue)) {
        try_pop_job();
    }

    {
    task_node* this_task = head_task.next;
    while(this_task != NULL) {
        task_node* next_task = this_task->next;
        free(this_task->task);
        free(this_task);
        this_task = next_task;
    }
    head_task.next = NULL;
    }

    pthread_mutex_destroy(&(pool.queue_lock));

    pool.threads = calloc(sizeof(pthread_t), n_merge_threads);
    pool.n_threads = n_merge_threads;
    pool.work_tail = &(pool.work_queue);
    pool.die = 0;

    pthread_mutex_init(&(pool.queue_lock), NULL);

    for(int i = 0; i < n_merge_threads; i++) {
        pthread_create(&(pool.threads[i]), NULL, pool_task, NULL);
    }

    int outf_size = strlen(fnames_left[0]) + 20+20+1+4+1;
    char* fname_cpy = calloc(sizeof(char), strlen(fnames_left[0]) + 1);
    strncpy(fname_cpy, fnames_left[0], strlen(fnames_left[0]));
    char* final_prefix = calloc(sizeof(char), strlen(fnames_left[0]) + 1);
    char* final_ptr = final_prefix;
    if(fname_cpy[0] == '/') {
        final_prefix[0] = '/';
        final_ptr++;
    }

    char* prefix = strtok(fname_cpy, "/");
    char* prev_prefix = NULL;
    while(prefix != NULL) {
        if(prev_prefix != NULL) {
            long lgt = strlen(prev_prefix);
            strncpy(final_ptr, prev_prefix, lgt);
            final_ptr += lgt;
            final_ptr[0] = '/';
            final_ptr++;
        }
        prev_prefix = prefix;
        prefix = strtok(NULL, "/");
    }

    while(files_left > 1) {
        this_task = &head_task;
        char** new_files = calloc(sizeof(char*), (files_left + 1) >> 1);

        for(int i = 0; i < files_left - (files_left % 2); i += 2) {
            new_files[i >> 1] = calloc(sizeof(char), outf_size);
            sprintf(new_files[i >> 1], "%s%d_%d", final_prefix, i, files_left);
            this_task->next = calloc(sizeof(task_node), 1);
            this_task = this_task->next;
            this_task->task = calloc(sizeof(task), 1);

            this_task->task->payload = calloc(sizeof(merge_payload), 1);
            ((merge_payload*)(this_task->task->payload))->fname1 = fnames_left[i];
            ((merge_payload*)(this_task->task->payload))->fname2 = fnames_left[i+1];
            ((merge_payload*)(this_task->task->payload))->fname_out = new_files[i >> 1];
            ((merge_payload*)(this_task->task->payload))->size = size;
            ((merge_payload*)(this_task->task->payload))->batch_size = batch_size;

            this_task->task->type = JOB_MERGE;
            push_job(this_task->task);
        }

        for(;;) {
            this_task = &head_task;
            char done = 1;
            while(this_task != NULL) {
                done = done && this_task->task->done;
                if(this_task->task->done && (this_task->task->payload != NULL)) {
                    free(this_task->task->payload);
                    this_task->task->payload = NULL;
                }
                this_task = this_task->next;
            }
            if(done) { break; }
            sleep(0.1);
        }

        long n_curr_files = files_left;
        if((files_left > 1) && (files_left % 2 != 0)) { // If off by one, just copy the last fname across to the next round
            // this also requires adjusting the resulting `files_left' for the next round.
            int tgt_idx = ((files_left-1) >> 1);
            long last_size = strlen(fnames_left[files_left-1]);
            new_files[tgt_idx] = calloc(sizeof(char), last_size+1);
            strncpy(new_files[tgt_idx], fnames_left[files_left-1], last_size);
            files_left += 2; // so that after division, we're +1 to account for the last file   
        }
        files_left >>= 1;

        for(int i = 0; i < n_curr_files; i++) {
            free(fnames_left[i]);
        }
        free(fnames_left);
 
        fnames_left = new_files;
    }

    free(final_prefix);
    free(fname_cpy);

    char* final_name = make_fname(fnames_left[0], 0);
    char* final_idx = make_fname(fnames_left[0], 1);

    char* real_out = make_fname(out, 0);
    char* out_idx = make_fname(out, 1);

    rename(final_name, real_out);
    remove(final_name);
    rename(final_idx, out_idx);
    remove(final_idx);

    free(out_idx);
    free(real_out);
    free(final_idx);
    free(final_name);

    free(fnames_left[0]);
    free(fnames_left);

    pool.die = 1;
    for(int i = 0; i < pool.n_threads; i++) {
        pthread_join(pool.threads[i], NULL);
    }
    free(pool.threads);

    while(pool.work_tail != &(pool.work_queue)) {
        try_pop_job();
    }

    {
    task_node* this_task = head_task.next;
    while(this_task != NULL) {
        task_node* next_task = this_task->next;
        free(this_task->task);
        free(this_task);
        this_task = next_task;
    }
    head_task.next = NULL;
    }

    pthread_mutex_destroy(&(pool.queue_lock));

    free(head_task.task);

    return total_out_size;
}

int find_data_cmp_db(const void* key, const void* obj) {
    float a = *((float*)key);
    db* robj = (db*)obj;
    float b = robj->mass;
    return (a < b? 1 : (a == b? 0 : (b == 0? -2 : -1)));
}

void* bsearch_ineq(void* key, void* data, uint64_t n_elts, uint64_t elt_size, int(*cmp)(const void*,const void*), char greater_than) {
    uint64_t low_idx = 0;
    uint64_t high_idx = n_elts-1;
    uint64_t idx = 0;
    uint64_t prev_idx = -1;
    void* ret = NULL;

    while(prev_idx != idx) {
        prev_idx = idx;
        idx = (low_idx + high_idx + 1) >> 1;
        void* this = data + idx*elt_size;
        int cmp_this = cmp(key, this);

        if(cmp_this > 0) {
            if(greater_than) {
                ret = this;
            }
            high_idx = idx-1;
        } else if(cmp_this == 0) {
            ret = this;
            high_idx = idx-1;
            low_idx = idx+1;
        } else {
            if(!greater_than && cmp_this > -2) {
                ret = this;
            }
            low_idx = idx+1;
        }
    }

    return ret;
}

range find_data(char* fname, uint64_t batch_size, uint64_t size, float low, float high) {
    void* ptr = NULL;
    uint64_t idx = 0;
    int64_t start_idx = -1;
    int64_t end_idx = -1;

    do {
        uint64_t start = idx*size*batch_size;
        ret r = load(fname, start, size, batch_size, 0);

        ptr = bsearch_ineq((void*)&low, r.data, r.n / size, size, &find_data_cmp_db, 1);
        if(ptr) {
            start_idx = (ptr - r.data) / size + idx * batch_size - 1;
        }
        free(r.data);
        if((r.n < batch_size * size) || ptr) { break; }
        idx++;
    } while(1);

    if(!ptr) {
        return (range){ .start = -1, .end = -1 };       
    }

    ptr = NULL;
    void* prev_ptr = NULL;
    do {
        uint64_t start = idx*size*batch_size;

        ret r = load(fname, start, size, batch_size, 0);

        prev_ptr = ptr;
        ptr = bsearch_ineq((void*)&high, r.data, r.n / size, size, &find_data_cmp_db, 0);
        if(ptr) {
            end_idx = (ptr - r.data) / size + idx * batch_size + 1;
        }
        free(r.data);
        if((r.n < size*batch_size) || !ptr) { break; }
        idx++;
    } while(1);

    if(!prev_ptr) {
        return (range){ .start = -1, .end = -1 };
    }

    return (range){ .start = start_idx, .end = end_idx };
}

void score_data_to_str(char* out, score_ret score_data, int idx, int max_size) {
    int printed_size = 0;
    printed_size += sprintf(out, "{'score':%f,'sumI':%f,'total_matched':%d,'theoretical':[", score_data.scores[idx], score_data.sumI[idx], score_data.total_matched[idx]);
    char early_out = 0;
    for(int j = 0; j < score_data.npeaks; j += 4) {
        int i = idx * 2 * score_data.npeaks + j * 2;
        if(score_data.theoretical[i] == 0) { early_out = 1; break; }
        int n = snprintf(out + printed_size, max_size - printed_size, "[%f,%f],[%f,%f],[%f,%f],[%f,%f]", score_data.theoretical[i], score_data.theoretical[i+1], score_data.theoretical[i+2], score_data.theoretical[i+3], score_data.theoretical[i+4], score_data.theoretical[i+5], score_data.theoretical[i+6], score_data.theoretical[i+7]);
        printed_size += n;
        if(printed_size >= max_size-1) { return; } // space for last comma, hence -1
        if(j != score_data.npeaks-1) { sprintf(out + printed_size, ","); }
    }
    int leftovers = early_out? 0 : score_data.npeaks % 4;
    unsigned tgt_size = MAX(0, max_size - printed_size);
    switch(leftovers) {
        case 3:
            printed_size += snprintf(out + printed_size, tgt_size, "[%f,%f],[%f,%f],[%f,%f]]", score_data.theoretical[score_data.npeaks-6], score_data.theoretical[score_data.npeaks-5], score_data.theoretical[score_data.npeaks-4], score_data.theoretical[score_data.npeaks-3], score_data.theoretical[score_data.npeaks-2], score_data.theoretical[score_data.npeaks-1]); break;
        case 2:
            printed_size += snprintf(out + printed_size, tgt_size, "[%f,%f],[%f,%f]]", score_data.theoretical[score_data.npeaks-4], score_data.theoretical[score_data.npeaks-3], score_data.theoretical[score_data.npeaks-2], score_data.theoretical[score_data.npeaks-1]); break;
        case 1:
            printed_size += snprintf(out + printed_size, tgt_size, "[%f,%f]]", score_data.theoretical[score_data.npeaks-2], score_data.theoretical[score_data.npeaks-1]); break;
        default: // 0
            printed_size += snprintf(out + printed_size, tgt_size, "]"); break;
    }
    if(printed_size >= max_size) { return; }
    printed_size += snprintf(out + printed_size, max_size - printed_size, ",'spec':[");
    if(printed_size >= max_size) { return; }
    early_out = 0;
    for(int j = 0; j < score_data.npeaks; j += 4) {
        int i = idx * 2 * score_data.npeaks + j * 2;
        if(score_data.spec[i] == 0) { early_out = 1; break; }
        int n = snprintf(out + printed_size, max_size - printed_size, "[%f,%f],[%f,%f],[%f,%f],[%f,%f]", score_data.spec[i], score_data.spec[i+1], score_data.spec[i+2], score_data.spec[i+3], score_data.spec[i+4], score_data.spec[i+5], score_data.spec[i+6], score_data.spec[i+7]);
        printed_size += n;
        if(printed_size >= max_size-1) { return; } // space for last comma, hence -1
        if(j != score_data.npeaks-1) { sprintf(out + printed_size, ","); }
    }
    leftovers = early_out? 0 : score_data.npeaks % 4;
    tgt_size = MAX(0, max_size - printed_size);
    switch(leftovers) {
        case 3:
            printed_size += snprintf(out + printed_size, tgt_size, "[%f,%f],[%f,%f],[%f,%f]]", score_data.spec[score_data.npeaks-6], score_data.spec[score_data.npeaks-5], score_data.spec[score_data.npeaks-4], score_data.spec[score_data.npeaks-3], score_data.spec[score_data.npeaks-2], score_data.spec[score_data.npeaks-1]); break;
        case 2:
            printed_size += snprintf(out + printed_size, tgt_size, "[%f,%f],[%f,%f]]", score_data.spec[score_data.npeaks-4], score_data.spec[score_data.npeaks-3], score_data.spec[score_data.npeaks-2], score_data.spec[score_data.npeaks-1]); break;
        case 1:
            printed_size += snprintf(out + printed_size, tgt_size, "[%f,%f]]", score_data.spec[score_data.npeaks-2], score_data.spec[score_data.npeaks-1]); break;
        default: // 0
            printed_size += snprintf(out + printed_size, tgt_size, "]"); break;
    }
    if(printed_size >= max_size) { return; }
    printed_size += snprintf(out + printed_size, max_size - printed_size, ",'dist':[");
    if(printed_size >= max_size) { return; }
    early_out = 0;
    for(int j = 0; j < score_data.npeaks; j += 4) {
        int i = idx * score_data.npeaks + j;
        if(score_data.distances[i] == 0) { early_out = 1; break; }
        int n = snprintf(out + printed_size, max_size - printed_size, "%f,%f,%f,%f", score_data.distances[i], score_data.distances[i+1], score_data.distances[i+2], score_data.distances[i+3]);
        printed_size += n;
        if(printed_size >= max_size-1) { return; } // space for last comma, hence -1
        if(j != score_data.npeaks-1) { sprintf(out + printed_size, ","); }
    }
    leftovers = early_out? 0 : score_data.npeaks % 4;
    tgt_size = MAX(0, max_size - printed_size);
    switch(leftovers) {
        case 3:
            printed_size += snprintf(out + printed_size, tgt_size, "%f,%f,%f]", score_data.distances[score_data.npeaks-3], score_data.distances[score_data.npeaks-2], score_data.distances[score_data.npeaks-1]); break;
        case 2:
            printed_size += snprintf(out + printed_size, tgt_size, "%f,%f]", score_data.distances[score_data.npeaks-2], score_data.distances[score_data.npeaks-1]); break;
        case 1:
            printed_size += snprintf(out + printed_size, tgt_size, "%f]", score_data.distances[score_data.npeaks-1]); break;
        default: // 0
            printed_size += snprintf(out + printed_size, tgt_size, "]"); break;
    }
    if(printed_size >= max_size) { return; }
    printed_size += snprintf(out + printed_size, max_size - printed_size, "}");
}

res* make_res(double* scores, score_ret score_data, char* title, float mass, float rt, int charge, db* cands, int n) {
    res* out = calloc(n, sizeof(res));
    for(int i = 0; i < n; i++) {
        strncpy(out[i].title, title, strlen(title));
        strncpy(out[i].description, cands[i].description, strlen(cands[i].description));
        strncpy(out[i].seq, cands[i].sequence, strlen(cands[i].sequence));
        memcpy(out[i].modseq, cands[i].mods, 128*sizeof(float));
        out[i].length = strlen(cands[i].sequence);
        out[i].calc_mass = cands[i].mass;
        out[i].mass = mass;
        out[i].rt = rt;
        out[i].charge = charge;
        out[i].score = scores[i];
        score_data_to_str(out[i].score_data, score_data, i, sizeof(out[i].score_data) / sizeof(char));
    }
    return out;
}*/

void free_ret(ret r) {
    free(r.data);
}

void free_ptr(void* r) {
    free(r);
}

void free_score(score_ret* score) {
    free(score[0].distances);
    free(score[0].mask);
    free(score);
}

double ppm_dist(float q, float c) {
    // implementation fail in identipy: they use a double KO criterion,
    // the first of which is an absolute error, because they rely on the scipy kdtree implementation
    // which requires a fixed upper-bound. This is reproduced here ONLY to prove comparison to identipy,
    // and should be removed when possible.
    float fake = ((q - c) / 2000.0) * 100000.0;
    float real = ((q - c) / q) * 100000.0;
    return MAX(fake, real);
}

/*
def identipy_rnhs(cands, query):
    import scipy
    from scipy import spatial
    ret = []
    scores = []

    spectrum = numpy.asarray(query['spec'])
    mz_array = spectrum[:,0]
    charge = query['charge']
    qmass = query['mass']

    nterm = blackboard.config['search'].getfloat('nterm cleavage')
    cterm = blackboard.config['search'].getfloat('cterm cleavage')
    tree = scipy.spatial.cKDTree(mz_array.reshape((mz_array.size, 1)))
    acc_ppm = blackboard.config['search']['matching unit'] == 'ppm'
    acc = blackboard.config['search'].getfloat('peak matching tolerance')
*/
score_ret* rnhs(score_data d) {
    query* queries = d.q;
    void* cands = d.cands;
    int n_cands = d.n_cands;
    int npeaks = d.npeaks;
    float tol = d.tol;
    int n_series = d.n_series;
    char ppm = d.ppm;

    score_ret* r = calloc(sizeof(score_ret), n_cands);
    double* distances = calloc(n_series * n_cands * npeaks, sizeof(double));
    char* mask = calloc(n_series * n_cands * npeaks, sizeof(char));

    for(int i = 0; i < n_series * n_cands * npeaks; i++) {
        distances[i] = -1.0;
    }

    int32_t* indices = calloc(n_series * n_cands * npeaks, sizeof(int32_t));
    memset(indices, -1, n_series * n_cands * npeaks * sizeof(int32_t));
    uint32_t* prev_inds = calloc(n_series * n_cands, sizeof(uint32_t));

/*
    for i in range(cands[1]):
*/
    for(int j = 0; j < n_cands; j++) {
/*
    norm = spectrum[:,1].sum()
*/
        double intens_norm = 0;
        query* q = &(((query*)queries)[j]);

        for(int i = 0; i < q->npeaks; i++) {
            intens_norm += q->spec[i*2+1];
        }
/*
        cand = cands[0][i]
        theoretical = numpy.zeros((cand.npeaks,)) # xxx: needs to be dict, etc.
        for j in range(cand.npeaks):
            theoretical[j] = cand.spec[j][0]

        score = 0
        mult = []
        match = {}
        match2 = {}
        total_matched = 0
        sumI = 0

        dist_all = []
*/
        db* cand = &(((db*)cands)[j]);

        r[j].distances = distances + npeaks * n_series * j;
        r[j].mask = mask + npeaks * n_series * j;

        uint64_t* mult = calloc(n_series, sizeof(uint64_t));

        double score = 0;
        uint32_t total_matched = 0;
        double total_intens = 0;

/*
        for ion, fragments in theoretical.items():
            if ion[-1] >= charge:
                break
 */
        for(int s = 0; s < n_series && cand->valid_series[s]; s++) {
/*
            dist, ind = tree.query(fragments, distance_upper_bound=acc if not acc_ppm else (float(acc) / 1e6 * 2000))
            
            mask1 = (dist != numpy.inf)
            if acc_ppm:
                nacc = numpy.where(dist[mask1] / mz_array[ind[mask1]] * 1e6 > acc)[0]
                mask2 = mask1.copy()
                mask2[nacc] = False
            else:
                mask2 = mask1
            nmatched = mask2.sum() // sum of: 1 if this query peak matched any theoretical peak, else 0
*/
            int candpeaks = cand->npeaks[s];
            int prev_ind = 0;
            int nmatched = 0;
            double intens_sum = 0;

            for(int k = 0; k < q->npeaks; k++) {
                for(int i = prev_ind; i < candpeaks; i++) {
                    double this_dist = ppm? ppm_dist(q->spec[k*2+0], cand->spec[s*npeaks+i]) : (q->spec[k*2+0] - cand->spec[s*npeaks+i]);
                    double abs_dist = ABS(this_dist);

                    if((abs_dist > tol) && (this_dist > 0)) {
                        prev_ind = i;
                    } else if(abs_dist <= tol) {
                        if((r[j].distances[s * npeaks + k] < 0) || (abs_dist < r[j].distances[s * npeaks + k])) {
                            r[j].distances[s * npeaks + k] = abs_dist;
                            indices[s * npeaks * n_cands + j * npeaks + k] = i;
                        }
                    } else if(this_dist < 0) {
                        // outside tol and higher m/z than current spec peak, impossible to find a better match
                        break;
                    }
                }

                if(r[j].distances[s * npeaks + k] > -1) {
                    intens_sum += q->spec[k*2+1];
                    r[j].mask[s * npeaks + k] = 1;
                    nmatched++;
                }
            }

/*
            if nmatched:
                total_matched += nmatched
                mult.append(numpy.math.factorial(nmatched))
                sumi = spectrum[:,1][ind[mask2]].sum()
                sumI += sumi
                score += sumi / norm
                dist_all.extend(dist[mask2])
            match[ion] = mask2
            match2[ion] = mask2
*/

            if(nmatched > 0) {
                total_matched += nmatched;
                total_intens += intens_sum;
                score += (float)(intens_sum) / intens_norm;
                mult[s] = FACT[MIN(nmatched, 20)];
            } else {
                mult[s] = 1;
            }
        }
/*
        if not total_matched:
            ret.append(None)
            scores.append(0)
            continue
*/
        if(total_matched == 0) {
            r[j].score = 0;
            free(mult);
            continue;
        }
/*
        for m in mult:
            score *= m
        sumI = numpy.log10(sumI)
*/

        for(int i = 0; i < n_series && cand->valid_series[i]; i++) {
            score *= mult[i]; 
        }
/*
        ret.append({'score': score, 'theoretical': theoretical, 'spec': mz_array.tolist(), 'match2': {k:v.tolist() for k, v in match2.items()}, 'match': {k:v.tolist() for k, v in match.items()}, 'sumI': sumI, 'dist': dist_all, 'total_matched': total_matched})
        scores.append(score)
*/
        //r[j].score = score;
        r[j].score = score;
        r[j].sumI = (total_intens != 0)? log10(total_intens) : 0;
        r[j].total_matched = total_matched;

        free(mult);
    }

    free(indices);
    free(prev_inds);
/*
    return scores, ret
*/
    return r;
}

//score_ret rnhs(query* q, void* cands, int n_cands, int npeaks, uint64_t elt_size, float tol) {
void* alloc(uint64_t n) {
    return calloc(1, n);
}

/*
int print_char_array(char** str, int offset, int* capacity, char* arr, int n_elts) {
    int orig_offset = offset;
    char* fmt_str = "%d,%d,%d,%d";
    char* fmt_str3 = "%d,%d,%d";
    char* fmt_str2 = "%d,%d";
    for(int i = 0; i < n_elts; i += 4) {
        if(i != 0) {
            offset += snprintf(*str + offset, *capacity - offset, ",");
        }
        int new_offset = snprintf(*str + offset, *capacity - offset, fmt_str, arr[i], arr[i+1], arr[i+2], arr[i+3]);
        if(new_offset >= *capacity - offset) {
            *capacity = MAX(2 * *capacity, *capacity - offset + new_offset);
            char* new_str = calloc(sizeof(char), *capacity);
            strcpy(new_str, *str);
            free(*str);
            *str = new_str;
            new_offset = snprintf(*str + offset, *capacity - offset, fmt_str, arr[i], arr[i+1], arr[i+2], arr[i+3]);
        }
        offset += new_offset;
    }

    if(n_elts % 4 != 0) {
        offset += snprintf(*str + offset, *capacity - offset, ",");
    }

    char done = 0;
    int new_offset = 0;
    while(!done) {
        switch(n_elts % 4) {
            case 3:
                new_offset = snprintf(*str + offset, *capacity - offset, fmt_str3, arr[n_elts-3], arr[n_elts-2], arr[n_elts-1]);
                break;
            case 2:
                new_offset = snprintf(*str + offset, *capacity - offset, fmt_str2, arr[n_elts-2], arr[n_elts-1]);
                break;
            case 1:
                new_offset = snprintf(*str + offset, *capacity - offset, "%d", arr[n_elts-1]);
                break;
            default: break; /* 0 /
        }
        if(new_offset >= *capacity - offset) {
            *capacity = MAX(2 * *capacity, *capacity - offset + new_offset);
            char* new_str = calloc(sizeof(char), *capacity);
            strcpy(new_str, *str);
            free(*str);
            *str = new_str;
        } else {
            done = 1;
            offset += new_offset;
        }
    }
    return offset - orig_offset;
}

int print_double_array(char** str, int offset, int* capacity, double* arr, int n_elts) {
    int orig_offset = offset;
    char* fmt_str = "%f,%f,%f,%f";
    char* fmt_str3 = "%f,%f,%f";
    char* fmt_str2 = "%f,%f";
    for(int i = 0; i < n_elts; i += 4) {
        if(i != 0) {
            offset += snprintf(*str + offset, *capacity - offset, ",");
        }
        int new_offset = snprintf(*str + offset, *capacity - offset, fmt_str, arr[i], arr[i+1], arr[i+2], arr[i+3]);
        if(new_offset >= *capacity - offset) {
            *capacity = MAX(2 * *capacity, *capacity - offset + new_offset);
            char* new_str = calloc(sizeof(char), *capacity);
            strcpy(new_str, *str);
            free(*str);
            *str = new_str;
            new_offset = snprintf(*str + offset, *capacity - offset, fmt_str, arr[i], arr[i+1], arr[i+2], arr[i+3]);
        }
        offset += new_offset;
    }

    if(n_elts % 4 != 0) {
        offset += snprintf(*str + offset, *capacity - offset, ",");
    }

    char done = 0;
    int new_offset = 0;
    while(!done) {
        switch(n_elts % 4) {
            case 3:
                new_offset = snprintf(*str + offset, *capacity - offset, fmt_str3, arr[n_elts-3], arr[n_elts-2], arr[n_elts-1]);
                break;
            case 2:
                new_offset = snprintf(*str + offset, *capacity - offset, fmt_str2, arr[n_elts-2], arr[n_elts-1]);
                break;
            case 1:
                new_offset = snprintf(*str + offset, *capacity - offset, "%f", arr[n_elts-1]);
                break;
            default: break; /* 0 /
        }
        if(new_offset >= *capacity - offset) {
            *capacity = MAX(2 * *capacity, *capacity - offset + new_offset);
            char* new_str = calloc(sizeof(char), *capacity);
            strcpy(new_str, *str);
            free(*str);
            *str = new_str;
        } else {
            done = 1;
            offset += new_offset;
        }
    }
    return offset - orig_offset;
}

char* score_str(score_ret* data, int n) {
    int max_size = 10240;
    char* ret = calloc(sizeof(char), max_size);
    int printed = snprintf(ret, max_size, "{'score':%f,'mask':[", data->scores[n]);
    printed += print_char_array(&ret, printed, &max_size, data->mask + (data->npeaks * n), data->npeaks);
    printed += snprintf(ret + printed, max_size - printed, "],'distances':[");
    printed += print_double_array(&ret, printed, &max_size, data->distances + (data->npeaks * n), data->npeaks);
    printed += snprintf(ret + printed, max_size - printed, "],'total_matched':%d,'sumI':%f}", data->total_matched[n], data->sumI[n]);
    return ret;
}
*/
