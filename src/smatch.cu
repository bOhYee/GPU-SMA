#include <stdio.h>
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
        //printf("Thread: %d\tIndex: %d", index, text_index);

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

        // Prepare next text window hash
        if (i < text_size - pattern_size)
            hash_text = (ALPHABET_SIZE * (hash_text - text[text_index+i] * h) + text[text_index+i+pattern_size]) % OVERFLOW_PM;

        if (hash_text < 0)
            hash_text += OVERFLOW_PM;
    }
}