#include <stdio.h>
#include "../inc/constants.h"
#include "../inc/smatch.h"

/* Rabin-Karp algorithm for CPU execution
 * Makes use of hashes to easily verify if two strings are equal
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
 * Every thread scans a string of length m to see if there are matches
 *
 * naive: no shared memory involved
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
 * Every thread scans a string of length m to see if there are matches
 *
 * Memory sharing is involved for optimization purposes
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
 * Used by the KMP algorithm to avoid useless comparisons between operands
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
 * Makes use of the concept of the Longest Proper Suffix to avoid returning to the previous index when
 * a miss verifies during the scan
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
 * Every thread manages a substring of the text to scan
 *
 * naive: no memory optmization involved
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
 * Every thread manages a substring of the text to scan
 *
 * Memory optimizations involved
 * Shared memory used for pattern and lps access time reduction
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



/* Bad character rule implementation for Boyer-Moore algorithm
 */
void bad_char_rule(int *bshifts, unsigned char *pattern, int pattern_size) {

    for (int i = 0; i < ALPHABET_SIZE; i++)
        bshifts[i] = -1;

    for (int i = 0; i < pattern_size - 1; i++) 
        bshifts[(int) pattern[i]] = pattern_size - 1 - i;

}


/* Check if a certain string is a prefix
 * Returns true if the suffix of pattern starting from pattern[pos] is also a prefix of pattern
 */ 
int is_prefix(unsigned char *pattern, int pattern_size, int pos) {

    int suffixlen = pattern_size - pos;

    for (int i = 0; i < suffixlen; i++) 
        if (pattern[i] != pattern[pos+i])
            return 0;
            
    return 1;
}


/* Returns the length of the longest suffix of pattern ending on pattern[pos]
 */ 
int suffix_length(unsigned char *pattern, int pattern_size, int pos) {

    int i;
    
    // Increment suffix length i to the first mismatch or beginning of the word
    for (i = 0; (pattern[pos-i] == pattern[pattern_size-1-i]) && (i < pos); i++);
    
    return i;
}


/* Good suffix rule implementation for Boyer-Moore algorithm
 * Strong version, as implemented by Dan Gusfield
 */
void good_suffix_rule(int *gshifts, unsigned char *pattern, int pattern_size) {

    int slen;
    int last_prefix_index = 1;

    // Prefix pattern
    for (int p = pattern_size - 1; p >= 0; p--) {
        if (is_prefix(pattern, pattern_size, p+1))
            last_prefix_index = p+1;

        gshifts[p] = (pattern_size-1 - p) + last_prefix_index;
    }

    // Suffix pattern
    for (int p = 0; p < pattern_size - 1; p++) {
        slen = suffix_length(pattern, pattern_size, p);

        if (pattern[p - slen] != pattern[pattern_size-1 - slen])
            gshifts[pattern_size-1 - slen] = pattern_size-1 - p + slen;
    }

}


/* Boyer-Moore algorithm for string matching
 * Tries to skip as many characters as possible by following two different rules:
 *     - bad character rule;
 *     - good suffix rule.
 */
void boyer_moore_cpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, 
                      int *bshifts, int *gshifts, int *match_result) {

    int i, j;
    i = pattern_size - 1;

    // Preprocessing of the pattern string
    bad_char_rule(bshifts, pattern, pattern_size);
    good_suffix_rule(gshifts, pattern, pattern_size);
    
    // Search
    while (i < text_size) {

        j = pattern_size-1;

        while (j >= 0 && (text[i] == pattern[j])) {
            --i;
            --j;
        }

        if (j < 0) {
            match_result[++i] = 1;
            i += gshifts[0];
        }
        else {
            if (bshifts[text[i]] < gshifts[j])
                i += gshifts[j];
            else
                i += bshifts[text[i]];
        }
    }
    
}


/* Naive version of the Boyer-Moore algorithm for string matching
 * Tries to skip as many characters as possible by following two different rules:
 *     - bad character rule;
 *     - good suffix rule.
 *
 * Naive implementation of the algorithm
 * No memory optimization
 */
__global__ void naive_boyer_moore_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, 
                                       int *bshifts, int *gshifts, int search_size, int *match_result) {

    int i, j, prev_i;
    unsigned int index, pos_int_block, block_pos_grid;
    unsigned int text_index;

    // Thread index
    pos_int_block = threadIdx.x + threadIdx.y * blockDim.x;
    block_pos_grid = (blockIdx.y * gridDim.x) + blockIdx.x;
    index = pos_int_block + block_pos_grid * blockDim.y * blockDim.x;
    text_index = search_size * index;

    // Search
    i = text_index + pattern_size - 1;
    while ((i >= text_index + pattern_size - 1) && (i < (text_index + search_size + pattern_size - 1)) && (i < text_size)) {

        j = pattern_size - 1;

        while (j >= 0 && (text[i] == pattern[j])){
            --j;
            --i;
        }

        if (j < 0) {
            match_result[++i] = 1;
            i += gshifts[0];
        }
        else {
            prev_i = i;

            if (bshifts[text[i]] < gshifts[j])
                i += gshifts[j];
            else
                i += bshifts[text[i]];

            // Avoid returning back to not encounter recursion over the same indexes
            if (i <= prev_i)
                i += pattern_size - 1;
        }
    }
}


/* Boyer-Moore algorithm for string matching
 * Tries to skip as many characters as possible by following two different rules:
 *     - bad character rule;
 *     - good suffix rule.
 *
 * Implementation makes use of shared memory for storing pattern, the bad rule table and the good rule table
 * Shared memory per block used = 5 * MAX_PATTERN_LENGTH + 4 * ALPHABET_SIZE bytes
 * It might be too much for some architectures: in that case lowering the constants should help
 */
__global__ void boyer_moore_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, 
                                 int *bshifts, int *gshifts, int search_size, int *match_result) {

    __shared__ unsigned char local_pattern[MAX_PATTERN_LENGTH];
    __shared__ unsigned int  local_bshifts[ALPHABET_SIZE];
    __shared__ unsigned int  local_gshifts[MAX_PATTERN_LENGTH];

    int i, j, copy_amount, start_over_shift, copy_index, prev_i;
    unsigned int index, pos_int_block, block_pos_grid;
    unsigned int text_index;

    // Thread index
    pos_int_block = threadIdx.x + threadIdx.y * blockDim.x;
    block_pos_grid = (blockIdx.y * gridDim.x) + blockIdx.x;
    index = pos_int_block + block_pos_grid * blockDim.y * blockDim.x;
    text_index = search_size * index;

    // Copy to shared memory the pattern and the two tables
    copy_amount = ceil(pattern_size / (blockDim.x * blockDim.y)) + 1;
    local_bshifts[pos_int_block] = bshifts[pos_int_block];

    for (int m = 0; m < copy_amount; m++) {
        copy_index = (index * copy_amount + m) % pattern_size;
        local_pattern[copy_index] = pattern[copy_index];
        local_gshifts[copy_index] = gshifts[copy_index];
        //printf("Thread: %d\tBlockX: %d\tBlockY: %d\tCI: %d\t%d\n", index, blockIdx.x, blockIdx.y, copy_index, local_gshifts[copy_index]);
    }
    __syncthreads();
    
    i = text_index + pattern_size - 1;
    start_over_shift = local_gshifts[0];

    // Search
    while ((i >= text_index + pattern_size - 1) && (i < (text_index + search_size + pattern_size - 1)) && (i < text_size)) {

        j = pattern_size - 1;

        while (j >= 0 && (text[i] == local_pattern[j])){
            --j;
            --i;
        }

        if (j < 0) {
            match_result[++i] = 1;
            i += start_over_shift;
        }
        else {
            prev_i = i;

            if (local_bshifts[text[i]] < local_gshifts[j])
                i += local_gshifts[j];
            else
                i += local_bshifts[text[i]];

            // Avoid returning back to not encounter recursion over the same indexes
            if (i <= prev_i)
                i += pattern_size - 1;
        }
    }
}