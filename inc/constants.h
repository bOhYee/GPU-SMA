/* Define if the program has to be launched through scripts or not
 */
#define SCRIPT_MODE 0

/* Location where the pattern to search is defined/stored
 *
 * 1 - Constant memory; 2 - Read from stdin or defined in main function
 */
#define PATTERN_LOCATION 2

/* Maximum pattern number
 * 
 * Used to define maximum amount of pattern that can be searched concurrently 
 * Statically defined to maximize throughput
 */
#define MAX_PATTERN_NUMBER 8

/* Maximum pattern length for string matching
 *
 * Needed for defining how much storage to prepare inside the shared memory
 */
#define MAX_PATTERN_LENGTH 800

/* Granularity represents how long the sequence each thread has to scan is
 *
 * It affects how much a single thread work
 * Its value also affects the parallelization factor of the GPU
 * 
 * Constraint: it should be at least >= 1, otherwise no thread works
 */ 
#define GRANULARITY_RK  100
#define GRANULARITY_KMP 100
#define GRANULARITY_BM  100

/* GPU parameters definition
 */
#define BLOCK_DIMENSION 16
#define MAX_NUM_STREAM 8
#define MAX_MULTIPT_STREAM 8

/* Algorithms that can be chosen for test runs
 */
enum algo { NAIVE_RK = 1, RK = 2, NAIVE_KMP = 3, KMP = 4, NAIVE_BM = 5, BM = 6 };