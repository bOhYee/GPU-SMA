/* General parameters
 */
#define ALPHABET_SIZE 256       // Alphabet size for every algorithm

/* Parameters for the Rabin-Karp algorithm
 */
#define OVERFLOW_PM 101         // Prime number used for overflow containment operations

/* Rabin-Karp */
/* Rabin-Karp algorithm for CPU execution
 * Makes use of hashes to easily verify if two strings are equal
 */
void rk_cpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *match_result);

/* Simple string matching algorithm
 * Every thread scans a string of length m to see if there are matches
 *
 * naive: no shared memory involved
 */
__global__ void naive_rk_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int search_size, int *match_result);

/* Simple string matching algorithm
 * Every thread scans a string of length m to see if there are matches
 *
 * Memory sharing is involved for optimization purposes
 */
__global__ void rk_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int search_size, int *match_result);


/* KMP */
/* Longest Proper Suffix calculation function
 * Used by the KMP algorithm to avoid useless comparisons between operands
 */
void compute_lps (unsigned char *pattern, int pattern_size, int *lps);

/* Knuth-Morris-Pratt algorithm for CPU execution
 * Makes use of the concept of the Longest Proper Suffix to avoid returning to the previous index when
 * a miss verifies during the scan
 */
void kmp_cpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *lps, int *match_result);

/* Simple KMP implementation for GPU execution
 * Every thread manages a substring of the text to scan
 *
 * naive: no memory optmization involved
 */
__global__ void naive_kmp_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *lps, int search_size, int *match_result);

/* Simple KMP implementation for GPU execution
 * Every thread manages a substring of the text to scan
 *
 * Memory optimizations involved
 * Shared memory used for pattern and lps access time reduction
 */
__global__ void kmp_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *lps, int search_size, int *match_result);


/* Boyer-Moore */
/* Bad character rule implementation for Boyer-Moore algorithm
 */
void bad_char_rule(int *bshifts, unsigned char *pattern, int pattern_size);

/* Check if a certain string is a prefix
 * Returns true if the suffix of pattern starting from pattern[pos] is also a prefix of pattern
 */ 
int is_prefix(unsigned char *pattern, int pattern_size, int pos);

/* Returns the length of the longest suffix of pattern ending on pattern[pos]
 */ 
int suffix_length(unsigned char *pattern, int pattern_size, int pos);

/* Good suffix rule implementation for Boyer-Moore algorithm
 * Strong version, as implemented by Dan Gusfield
 */
void good_suffix_rule(int *gshifts, unsigned char *pattern, int pattern_size);

/* Boyer-Moore algorithm for string matching
 * Tries to skip as many characters as possible by following two different rules:
 *     - bad character rule;
 *     - good suffix rule.
 */
void boyer_moore_cpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *bshifts, int *gshifts, int *match_result);

/* Naive version of the Boyer-Moore algorithm for string matching
 * Tries to skip as many characters as possible by following two different rules:
 *     - bad character rule;
 *     - good suffix rule.
 *
 * Naive implementation of the algorithm
 * No memory optimization
 */
__global__ void naive_boyer_moore_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *bshifts, int *gshifts, int search_size, int *match_result);

/* Boyer-Moore algorithm for string matching
 * Tries to skip as many characters as possible by following two different rules:
 *     - bad character rule;
 *     - good suffix rule.
 *
 * Implementation makes use of shared memory for storing pattern, the bad rule table and the good rule table
 * Shared memory per block used = 5 * MAX_PATTERN_LENGTH + 4 * ALPHABET_SIZE bytes
 * It might be too much for some architectures: in that case lowering the constants should help
 */
__global__ void boyer_moore_gpu (unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *bshifts, int *gshifts, int search_size, int *match_result);