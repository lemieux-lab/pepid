#include "helper.h"

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

uint64_t dump(char* fname, uint64_t file_offset, void* data, uint64_t total_size, char erase) {
    FILE* outf = fopen(fname, erase? "wb" : "r+b");
    fseeko(outf, file_offset, SEEK_SET);
    printf("DUMP! %s %llu %p %llu\n", fname, total_size, data, file_offset);
    printf("D> %f\n", *((float*)(data + total_size - 4)));
    uint64_t written = fwrite(data, 1, total_size, outf);
    printf("DUMPED, %llu!\n", written);
    fclose(outf);
    return written;
}

ret load(char* fname, uint64_t offset, uint64_t size, uint64_t start) {
    FILE* inf = fopen(fname, "rb");

    struct stat s;
    fstat(fileno(inf), &s);

    uint64_t size_to_read = (size == 0? s.st_size - offset:MIN(size,s.st_size - offset));

    fseeko(inf, offset, SEEK_SET);
    void* out = calloc(1, size_to_read);
    uint64_t actually_read = fread(out, 1, size_to_read, inf);

    fclose(inf);

    return (ret){ .n = actually_read, .data = out, .start = offset + start };
}

void dump_buffer(db_buffer* buffer) {
    // idx marks 1 past the last index to dump.
    if((buffer->in->idx > buffer->mark) && ((buffer->in->idx - buffer->start) <= buffer->end)) {
        printf("dump_buffer: ");
        printf("%s ", buffer->out->fname);
        printf("%llu x %llu ", buffer->out->idx, buffer->elt_size);
        printf("[ %p + ( %llu - %llu + 1 ) ", buffer->data, buffer->mark, buffer->start);
        printf(" ( / %llu )) * %llu ]\n", buffer->end, buffer->elt_size);
        dump(buffer->out->fname, buffer->out->idx*buffer->elt_size, buffer->data + (buffer->mark - buffer->start) * buffer->elt_size, (buffer->in->idx - buffer->mark + 1) * buffer->elt_size,  0);
        printf("... dumped data\n");
        printf("-> %s %llu (%llu) %p %llu %llu %llu\n", buffer->out->fname_idx, buffer->out->idx, buffer->idx_elt_size, buffer->data_idx, buffer->mark, buffer->start, buffer->end);
        dump(buffer->out->fname_idx, buffer->out->idx*buffer->idx_elt_size, buffer->data_idx + (buffer->mark - buffer->start) * buffer->idx_elt_size, (buffer->in->idx - buffer->mark + 1) * buffer->idx_elt_size,  0);
        printf("... dumped indices\n");
        buffer->out->idx += (buffer->in->idx - buffer->mark + 1);
        buffer->mark = buffer->in->idx + 1;
    }
}

void buffer_next(db_buffer* buffer) {
    buffer->done = (buffer->end > 0) && ((buffer->end - buffer->start) < buffer->batch_size) && (buffer->in->idx >= buffer->end);
    if(!(buffer->done)) {
        if(buffer->in->idx >= buffer->end) {
            dump_buffer(buffer);
            ret r = load(buffer->in->fname, buffer->in->idx*buffer->elt_size, buffer->elt_size*buffer->batch_size, buffer->in->start);
            ret r_idx = load(buffer->in->fname_idx, buffer->in->idx*buffer->idx_elt_size, buffer->idx_elt_size*buffer->batch_size, buffer->in->start);
            printf("LATEST IDX: %llu %llu\n", r_idx.n, r_idx.n / sizeof(uint64_t));
            free(buffer->data);
            free(buffer->data_idx);
            buffer->data = r.data;
            buffer->data_idx = r_idx.data;
            buffer->start = buffer->in->idx;
            buffer->end = (r.n / buffer->elt_size) + buffer->start;
        } else {
            buffer->in->idx++;
        } 
    } else {
        return;
    }
}

void merge_sort_merge(char* fname1, char* fname2, uint64_t start1, uint64_t start2, char* out, uint32_t size, uint32_t batch_size) {
    printf("%s + %s -> %s (%d, %d x %d (%d))\n", fname1, fname2, out, start1, start2, size, batch_size);
    int lgt1 = strlen(fname1);
    char* full_name1 = calloc(lgt1 + 5, 1);
    sprintf(full_name1, "%s.bin", fname1);
    char* idx_name1 = calloc(lgt1 + 4 + 5, 1);
    sprintf(idx_name1, "%s_idx.bin", fname1);

    int lgt2 = strlen(fname2);
    char* full_name2 = calloc(lgt2 + 5, 1);
    sprintf(full_name2, "%s.bin", fname2);
    char* idx_name2 = calloc(lgt2 + 4 + 5, 1);
    sprintf(idx_name2, "%s_idx.bin", fname2);

    int lgt_out = strlen(out);
    char* full_name_out = calloc(lgt_out + 5, 1);
    sprintf(full_name_out, "%s.bin", out);
    char* idx_name_out = calloc(lgt_out + 4 + 5, 1);
    sprintf(idx_name_out, "%s_idx.bin", out);

    FILE* outf = fopen(full_name_out, "wb");
    FILE* idx_outf = fopen(idx_name_out, "wb");
    fclose(outf); // just create the file, will be filled later
    fclose(idx_outf); // just create the file, will be filled later

    printf("Got fnames, created files...\n");

    db_buffer buff1;
    db_buffer buff2;

    buff1.elt_size = size;
    buff1.idx_elt_size = sizeof(uint64_t);
    buff2.elt_size = size;
    buff2.idx_elt_size = sizeof(uint64_t);
    buff1.batch_size = batch_size;
    buff2.batch_size = batch_size;
    indexed_file in1 = (indexed_file){ .fname = full_name1, .fname_idx = idx_name1, .idx = 0, .start = start1 };
    indexed_file in2 = (indexed_file){ .fname = full_name2, .fname_idx = idx_name2, .idx = 0, .start = start2 };
    indexed_file out_obj = (indexed_file){ .fname = full_name_out, .fname_idx = idx_name_out, .idx = 0, .start = 0 };
    buff1.in = &in1;
    buff2.in = &in2;
    buff1.out = &out_obj;
    buff2.out = &out_obj;
    buff1.end = 0;
    buff2.end = 0;
    buff1.start = 0;
    buff2.start = 0;
    buff1.mark = 0;
    buff2.mark = 0;
    buff1.data = NULL;
    buff1.data_idx = NULL;
    buff2.data = NULL;
    buff2.data_idx = NULL;
    buff1.done = 0;
    buff2.done = 0;

    buffer_next(&buff1);
    buffer_next(&buff2);
    float this1 = ((float*)(buff1.data))[buff1.in->idx - buff1.start];
    float this2 = ((float*)(buff2.data))[buff2.in->idx - buff2.start];


    printf("task init done\n");

    while(!buff1.done || !buff2.done) {
        if(buff2.done) {
            while(!buff1.done) {
                printf("!buff1.done BEEP\n");
                buffer_next(&buff1);
            }
            dump_buffer(&buff1);
        } else if(buff1.done) {
            while(!buff2.done) {
                printf("!buff2.done BOOP\n");
                buffer_next(&buff2);
            }
            dump_buffer(&buff2);
        } else if(!buff1.done && !buff2.done) {
            printf("A1: %lld - %lld\n", buff1.in->idx, buff1.start);
            printf("A2: %lld - %lld\n", buff2.in->idx, buff2.start);
                printf("-> %f %f\n", this1, this2);
            buffer_next(&buff1);
            buffer_next(&buff2);
            printf("...next\n");

            while((!buff1.done) && (this1 <= this2)) {
                this1 = ((float*)(buff1.data))[buff1.in->idx - buff1.start];
                buffer_next(&buff1);
            }
            printf("dump1\n");
            dump_buffer(&buff1);

            while((!buff2.done) && (this2 < this1)) {
                this2 = ((float*)(buff2.data))[buff2.in->idx - buff2.start];
                buffer_next(&buff2);
            }
            printf("Ready dump2\n");
            dump_buffer(&buff2);
            printf("dump2\n");

            this1 = ((float*)(buff1.data))[buff1.in->idx - buff1.start];
            this2 = ((float*)(buff2.data))[buff2.in->idx - buff2.start];
        } else { break; }
    }

    printf("%s done merge!\n", buff1.out->fname);

    free(full_name_out);
    free(idx_name_out);
    free(full_name1);
    free(idx_name1);
    free(full_name2);
    free(idx_name2);

    free(buff1.data);
    free(buff2.data);
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
    int lgt = strlen(fname);
    char* full_name = calloc(lgt + 5, 1);
    sprintf(full_name, "%s.bin", fname);
    char* idx_name = calloc(lgt + 4 + 5, 1);
    sprintf(idx_name, "%s_idx.bin", fname);

    ret r = load(full_name, 0, 0, start);

    uint64_t base = r.start;
    uint64_t* idx_arr = calloc(r.n / size, sizeof(uint64_t));
    for(int i = 0; i < r.n / size; i++) {
        idx_arr[i] = base + i;
    }

    qsort_r(idx_arr, r.n / size, sizeof(uint64_t), &merge_argsort_cmp_db, (void*)(&r));
    reorder_key((float*)(r.data), idx_arr, r.n / size, base);

    uint64_t ret = dump(full_name, 0, r.data, r.n, 1);
    dump(idx_name, 0, idx_arr, (r.n / size) * sizeof(uint64_t), 1);

    free(r.data);
    free(idx_arr);
    free(idx_name);
    free(full_name);

    return ret;
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
    //int thread_num = (int)arg; // hack!
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
                    merge_sort_merge(payload->fname1, payload->fname2, payload->start1, payload->start2, payload->fname_out, payload->size, payload->batch_size);
                    char* full_fname1 = calloc(strlen(payload->fname1) + 1 + 4, 1);
                    char* full_fname2 = calloc(strlen(payload->fname2) + 1 + 4, 1);
                    char* full_idx1 = calloc(strlen(payload->fname1) + 1 + 4 + 4, 1);
                    char* full_idx2 = calloc(strlen(payload->fname2) + 1 + 4 + 4, 1);
                    sprintf(full_fname1, "%s.bin", payload->fname1);
                    sprintf(full_fname2, "%s.bin", payload->fname2);
                    sprintf(full_idx1, "%s_idx.bin", payload->fname1);
                    sprintf(full_idx2, "%s_idx.bin", payload->fname2);
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
    printf("mergesort\n");
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

        char* this_fname = calloc(lgt+1+4, sizeof(char));
        sprintf(this_fname, "%s.bin", fnames[i]);
        FILE* f = fopen(this_fname, "rb");
        fseeko(f, 0, SEEK_END);
        start += ftello(f);
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

    printf("preparing to merge\n");
    uint64_t leftovers_start = 0;
    char has_leftovers = 0;
    while(files_left > 1) {
        this_task = &head_task;
        char** new_files = calloc(sizeof(char*), (files_left + 1) >> 1);
        uint64_t start = 0;

        for(int i = 0; i < files_left - (files_left % 2); i += 2) {
            new_files[i >> 1] = calloc(sizeof(char), outf_size);
            sprintf(new_files[i >> 1], "%s%d_%d", final_prefix, i, files_left);
            this_task->next = calloc(sizeof(task_node), 1);
            this_task = this_task->next;
            this_task->task = calloc(sizeof(task), 1);

            int lgt = strlen(fnames_left[i]);
            char* bin_fname1 = calloc(sizeof(char), lgt+1+4);
            char* bin_fname2 = calloc(sizeof(char), lgt+1+4);
            sprintf(bin_fname1, "%s.bin", fnames_left[i]);
            sprintf(bin_fname2, "%s.bin", fnames_left[i+1]);

            FILE* f1 = fopen(bin_fname1, "rb");
            FILE* f2 = fopen(bin_fname2, "rb");

            fseeko(f1, 0, SEEK_END);
            fseeko(f2, 0, SEEK_END);
            uint64_t fsize1 = ftello(f1);
            uint64_t fsize2 = ftello(f2);

            fclose(f1);
            fclose(f2);

            free(bin_fname1);
            free(bin_fname2);

            this_task->task->payload = calloc(sizeof(merge_payload), 1);
            ((merge_payload*)(this_task->task->payload))->fname1 = fnames_left[i];
            ((merge_payload*)(this_task->task->payload))->fname2 = fnames_left[i+1];
            ((merge_payload*)(this_task->task->payload))->start1 = start;
            ((merge_payload*)(this_task->task->payload))->start2 = ((i == files_left - 2) && has_leftovers)? leftovers_start : start + fsize1 / size;
            ((merge_payload*)(this_task->task->payload))->fname_out = new_files[i >> 1];
            ((merge_payload*)(this_task->task->payload))->size = size;
            ((merge_payload*)(this_task->task->payload))->batch_size = batch_size;

            start += fsize2;

            this_task->task->type = JOB_MERGE;
            push_job(this_task->task);
        }
        printf("Merge tasks all pushed!\n");

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
            has_leftovers = 1;
            leftovers_start = start;
            int tgt_idx = ((files_left-1) >> 1);
            long last_size = strlen(fnames_left[files_left-1]);
            new_files[tgt_idx] = calloc(sizeof(char), last_size+1);
            strncpy(new_files[tgt_idx], fnames_left[files_left-1], last_size);
            files_left += 2; // so that after division, we're +1 to account for the last file   
        } else {
            has_leftovers = 0;
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

    int lgt = strlen(fnames_left[0]);
    char* final_name = calloc(lgt + 5, 1);
    sprintf(final_name, "%s.bin", fnames_left[0]);
    char* final_idx = calloc(lgt + 4 + 5, 1);
    sprintf(final_idx, "%s_idx.bin", fnames_left[0]);

    lgt = strlen(out);
    char* real_out = calloc(lgt + 5, 1);
    sprintf(real_out, "%s.bin", out);
    char* out_idx = calloc(lgt + 4 + 5, 1);
    sprintf(out_idx, "%s_idx.bin", out);

    rename(final_name, real_out);
    rename(final_idx, out_idx);

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
        uint64_t size_limit = size*batch_size;
        ret r = load(fname, start, size_limit, 0);

        ptr = bsearch_ineq((void*)&low, r.data, r.n / size, size, &find_data_cmp_db, 1);
        if(ptr) {
            start_idx = (ptr - r.data) / size + idx * batch_size - 1;
        }
        free(r.data);
        if(r.n < size_limit || ptr) { break; }
        idx++;
    } while(1);

    if(!ptr) {
        return (range){ .start = -1, .end = -1 };       
    }

    ptr = NULL;
    void* prev_ptr = NULL;
    do {
        uint64_t start = idx*size*batch_size;
        uint64_t size_limit = size*batch_size;

        ret r = load(fname, start, size_limit, 0);

        prev_ptr = ptr;
        ptr = bsearch_ineq((void*)&high, r.data, r.n / size, size, &find_data_cmp_db, 0);
        if(ptr) {
            end_idx = (ptr - r.data) / size + idx * batch_size + 1;
        }
        free(r.data);
        if(r.n < size_limit || !ptr) { break; }
        idx++;
    } while(1);

    if(!prev_ptr) {
        return (range){ .start = -1, .end = -1 };
    }

    return (range){ .start = start_idx, .end = end_idx };
}
