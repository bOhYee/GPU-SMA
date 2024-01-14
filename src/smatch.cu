#include <stdio.h>
#include "../inc/constants.h"
#include "../inc/smatch.h"

/* Rabin-Karp algorithm for CPU execution
*  Makes use of hashes to easily verify if two strings are equal
*/
void rk_cpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *match_result) {

    int hash_text, hash_pattern, h, found;

    // Hash the first text window and pattern
    hash_pattern = 0;
    hash_text = 0;
    for (int i = 0; i < pattern_size; i++) {
        hash_pattern = (ALPHABET_SIZE * hash_pattern + pattern[i]) % OVERFLOW_PM;
        hash_text = (ALPHABET_SIZE * hash_text + text[i]) % OVERFLOW_PM;
    }

    // Compute the highest pow of ALPHABET_SIZE
    h = 1;
    for (int i = 0; i < pattern_size-1; i++) 
        h = (h * ALPHABET_SIZE) % OVERFLOW_PM;

    // Search the string
    for (int i = 0; i < (text_size - pattern_size + 1); i++) {

        // Check hashes
        found = 0;
        if (hash_pattern == hash_text) {
            found = 1;

            for (int j = 0; j < pattern_size; j++) {
                if (pattern[j] != text[i + j]) {
                    found = 0;
                    break;
                }
            }
        }

        // Save result
        match_result[i] = found;

        // Compute next window's hash
        if (i < text_size - pattern_size) 
            hash_text = (ALPHABET_SIZE * (hash_text - text[i] * h) + text[i+pattern_size]) % OVERFLOW_PM;

        if (hash_text < 0)
            hash_text += OVERFLOW_PM;
    }
}


/* Simple string matching algorithm
*  Every thread scans a string of length m to see if there are matches
*
*  naive: no shared memory involved
*/
__global__ void naive_rk_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size,
                              int search_size, int *match_result) {

    unsigned int index, pos_int_block, block_pos_grid;
    unsigned int text_index;

    int hash_text, hash_pattern, h, found;

    pos_int_block = threadIdx.x + threadIdx.y * blockDim.x;
    block_pos_grid = (blockIdx.y * gridDim.x) + blockIdx.x;
    index = pos_int_block + block_pos_grid * blockDim.y * blockDim.x;
    text_index = search_size * index;

    // Hash the first text window and pattern
    hash_text = 0;
    hash_pattern = 0;
    for (int i = 0; i < pattern_size; i++) {
        hash_pattern = (ALPHABET_SIZE * hash_pattern + pattern[i]) % OVERFLOW_PM;
        hash_text = (ALPHABET_SIZE * hash_text + text[text_index + i]) % OVERFLOW_PM;
    }

    // Computation of h = ALPHABET_SIZE ^ (PATTERN_SIZE-1)
    h = 1;
    for (int i = 0; i < pattern_size-1; i++) 
        h = (h * ALPHABET_SIZE) % OVERFLOW_PM;

    for (int i = 0; (i < search_size) && ((text_index + i) < (text_size - pattern_size + 1)); i++) {
        //printf("Thread: %d\tIndex: %d\tText size: %d\tText start: %d\n", index, text_index, text_size, text);

        // If the hashes are equal, most likely a hit but a check is required
        found = 0;
        if (hash_pattern == hash_text) {
            found = 1;

            for (int j = 0; j < pattern_size; j++)
                if (pattern[j] != text[text_index + i + j])
                    found = 0;
        }

        // Save result
        match_result[text_index + i] = found;

        // if (found == 1)
        //     printf("Thread: %d\tIndex: %d\tText size: %d\tText start: %d\n", index, text_index + i, text_size, text);

        // Prepare next text window hash
        if (i < text_size - pattern_size)
            hash_text = (ALPHABET_SIZE * (hash_text - text[text_index+i] * h) + text[text_index+i+pattern_size]) % OVERFLOW_PM;

        if (hash_text < 0)
            hash_text += OVERFLOW_PM;
    }
}


/* Simple string matching algorithm
*  Every thread scans a string of length m to see if there are matches
*
*  Memory sharing is involved for optimization purposes
*/
__global__ void rk_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size,
                        int search_size, int *match_result) {

    __shared__ unsigned char local_pattern[MAX_PATTERN_LENGTH];
    unsigned int index, pos_int_block, block_pos_grid;
    unsigned int text_index;

    int hash_text, hash_pattern, h, found;
    int copy_amount, copy_index;

    pos_int_block = threadIdx.x + threadIdx.y * blockDim.x;
    block_pos_grid = (blockIdx.y * gridDim.x) + blockIdx.x;
    index = pos_int_block + block_pos_grid * blockDim.y * blockDim.x;
    text_index = search_size * index;

    // Copy to shared memory the pattern for better access times
    copy_amount = ceil(pattern_size / (blockDim.x * blockDim.y)) + 1;
    for (int m = 0; m < copy_amount; m++) {
        copy_index = (index * copy_amount + m) % pattern_size;
        local_pattern[copy_index] = pattern[copy_index];
        //printf("Thread: %d\tBlockX: %d\tBlockY: %d\tCI: %d\t%c\n", index, blockIdx.x, blockIdx.y, copy_index, local_pattern[copy_index]);
    }
    __syncthreads();

    // Hash the first text window and pattern
    hash_text = 0;
    hash_pattern = 0;
    for (int i = 0; i < pattern_size; i++) {
        hash_pattern = (ALPHABET_SIZE * hash_pattern + local_pattern[i]) % OVERFLOW_PM;
        hash_text = (ALPHABET_SIZE * hash_text + text[text_index + i]) % OVERFLOW_PM;
    }

    // Computation of h = ALPHABET_SIZE ^ (PATTERN_SIZE-1)
    h = 1;
    for (int i = 0; i < pattern_size-1; i++) 
        h = (h * ALPHABET_SIZE) % OVERFLOW_PM;

    for (int i = 0; (i < search_size) && ((text_index + i) < (text_size - pattern_size + 1)); i++) {
        //printf("Thread: %d\tIndex: %d\n", index, search_size);

        // If the hashes are equal, most likely a hit but a check is required
        found = 0;
        if (hash_pattern == hash_text) {
            found = 1;

            for (int j = 0; j < pattern_size; j++)
                if (local_pattern[j] != text[text_index + i + j])
                    found = 0;
        }

        // Save result
        match_result[text_index + i] = found;

        // Prepare next text window hash
        if (i < text_size - pattern_size)
            hash_text = (ALPHABET_SIZE * (hash_text - text[text_index+i] * h) + text[text_index+i+pattern_size]) % OVERFLOW_PM;

        if (hash_text < 0)
            hash_text += OVERFLOW_PM;
    }
}


/* Longest Proper Suffix calculation function
*  Used by the KMP algorithm to avoid useless comparisons between operands
*/
void compute_lps (unsigned char *pattern, int pattern_size, int *lps) {

    int i, sub_i;

    // Compute LPS values for the pattern string
    i = 1; 
    sub_i = 0;
    lps[0] = 0;

    while (i < pattern_size) {

        if (pattern[i] == pattern[sub_i]) {
            sub_i++;
            lps[i++] = sub_i;
        }
        else {
            if (sub_i != 0) {
                sub_i = lps[sub_i-1];
            }
            else {
                lps[i] = 0;
                i++;
            }
        }

    }
}


/* Knuth-Morris-Pratt algorithm for CPU execution
*  Makes use of the concept of the Longest Proper Suffix to avoid returning to the previous index when
*  a miss verifies during the scan
*/
void kmp_cpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *lps, int *match_result) {

    int i, j;
    compute_lps(pattern, pattern_size, lps);

    // Compare the strings now
    i = 0;
    j = 0;
    while ((text_size - i) >= (pattern_size - j)) {

        // Increase indexes if they keep matching
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        // When they don't match, try comparing the previous LPS string against the new character
        else if (i < text_size && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }

        /* When pattern is entirely evaluated against the text and a match has been found
        *  Move back to the previous LPS to avoid re-comparing old characters
        */
        if (j == pattern_size) {
            match_result[i-j] = 1;
            j = lps[j-1];
        } 
    }
}


/* Simple KMP implementation for GPU execution
*  Every thread manages a substring of the text to scan
*
*  naive: no memory optmization involved
*/
__global__ void naive_kmp_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *lps, 
                               int search_size, int *match_result) {

    int i, j;
    unsigned int index, pos_int_block, block_pos_grid;
    unsigned int text_index;

    // Thread index
    pos_int_block = threadIdx.x + threadIdx.y * blockDim.x;
    block_pos_grid = (blockIdx.y * gridDim.x) + blockIdx.x;
    index = pos_int_block + block_pos_grid * blockDim.y * blockDim.x;
    text_index = search_size * index;

    // Compare the strings now
    i = text_index;
    j = 0;
    while ((i < (text_index + search_size + pattern_size - 1)) && (text_size - i) >= (pattern_size - j)) {

        // Increase indexes if they keep matching
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        // When they don't match, try comparing the previous LPS string against the new character
        else if (i < text_size && pattern[j] != text[i]) {
            if (j != 0)
                j = lps[j - 1];
            else
                i = i + 1;
        }

        /* When pattern is entirely evaluated against the text and a match has been found
        *  Move back to the previous LPS to avoid re-comparing old characters
        */
        if (j == pattern_size) {
            printf("%d\t%d\n", index, text_index + i);
            match_result[i-j] = 1;
            j = lps[j-1];
        } 
    }
}


/* Simple KMP implementation for GPU execution
*  Every thread manages a substring of the text to scan
*
*  Memory optimizations involved
*  Shared memory used for pattern and lps access time reduction
*/
__global__ void kmp_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *lps, 
                         int search_size, int *match_result) {
                            
    __shared__ unsigned char local_pattern[MAX_PATTERN_LENGTH];
    __shared__ int local_lps[MAX_PATTERN_LENGTH];

    int i, j, copy_amount, copy_index;
    unsigned int index, pos_int_block, block_pos_grid;
    unsigned int text_index;

    // Thread index
    pos_int_block = threadIdx.x + threadIdx.y * blockDim.x;
    block_pos_grid = (blockIdx.y * gridDim.x) + blockIdx.x;
    index = pos_int_block + block_pos_grid * blockDim.y * blockDim.x;
    text_index = search_size * index;

    // Copy to shared memory both pattern and LPS table
    copy_amount = ceil(pattern_size / (blockDim.x * blockDim.y)) + 1;
    for (int m = 0; m < copy_amount; m++) {
        copy_index = (index * copy_amount + m) % pattern_size;
        local_pattern[copy_index] = pattern[copy_index];
        local_lps[copy_index] = lps[copy_index];
        //printf("Thread: %d\tBlockX: %d\tBlockY: %d\tCI: %d\t%c\n", index, blockIdx.x, blockIdx.y, copy_index, local_pattern[copy_index]);
    }
    __syncthreads();

    // Compare the strings now
    i = text_index;
    j = 0;
    while ((i < (text_index + search_size + pattern_size - 1)) && (text_size - i) >= (pattern_size - j)) {

        // Increase indexes if they keep matching
        if (local_pattern[j] == text[i]) {
            i++;
            j++;
        }
        // When they don't match, try comparing the previous LPS string against the new character
        else if (i < text_size && local_pattern[j] != text[i]) {
            if (j != 0)
                j = local_lps[j - 1];
            else
                i = i + 1;
        }

        /* When pattern is entirely evaluated against the text and a match has been found
        *  Move back to the previous LPS to avoid re-comparing old characters
        */
        if (j == pattern_size) {
            match_result[i-j] = 1;
            j = local_lps[j-1];
        } 
    }
}