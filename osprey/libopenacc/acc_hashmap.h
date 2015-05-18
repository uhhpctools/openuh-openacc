/**
 * Author: Rengan Xu
 * University of Houston
 */

#ifndef __ACC_HASHMAP_H
#define __ACC_HASHMAP_H

#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "acc_log.h"

typedef void* acc_hash;

typedef struct acc_hm_entry_s{
    int key;
    void *value;
    int hash;
    struct acc_hm_entry_s* next;
} acc_hm_entry;

typedef struct acc_hashmap_s{
    /* the number of key-value mappings in this hashmap */
    int size;

    int capacity;
    int load_factor;
    /*
     * threshold (=capacity*load_factor), the table is resized 
     * if its size is larger than this threshold
     */
    int threshold;

    acc_hm_entry** table;  
}acc_hashmap;

acc_hash acc_hashmap_create();

acc_hash acc_hashmap_create_with_args(int initial_capacity, float load_factor);

/* insert a key-map pair entry into the hashmap */
void* acc_hashmap_put(void* _hm, int key, void* value);

/* get a key-value entry from the hashmap based on the key value */
void* acc_hashmap_get(void* _hm, int key);

/* remove a key-value entry from the hashmap based on the key value */
void* acc_hashmap_remove(void* _hm, int key);

/* remove all key-value mappings in this hashmap */
void acc_hashmap_clear(void* _hm);

/* destroy the hashmap */
void acc_hashmap_destroy(void* _hm);

#endif
