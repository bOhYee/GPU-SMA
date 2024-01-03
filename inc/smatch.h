/* Parameters for the Rabin-Karp algorithm
*/
#define ALPHABET_SIZE 256       // Alphabet size for hashing-based algorithms
#define OVERFLOW_PM 101         // Prime number used for overflow containment operations

/* Rabin-Karp algorithm for CPU execution
*  Makes use of hashes to easily verify if two strings are equal
*/
void rk_cpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *match_result);

/* Simple string matching algorithm
*  Every thread scans a string of length m to see if there are matches
*
*  naive: no shared memory involved
*/
__global__ void naive_rk_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int search_size, int *match_result);

/* Knuth-Morris-Pratt algorithm for CPU execution
*  Makes use of the concept of the Longest Proper Suffix to avoid returning to the previous index when
*  a miss verifies during the scan
*/
void kmp_cpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *match_result);