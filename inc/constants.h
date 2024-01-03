/* Location where the pattern to search is defined/stored
*  1 - Constant memory; 2 - Read from stdin or defined in main function
*/
#define PATTERN_LOCATION 2

/* Pattern length
*  Used when this is statically defined
*/
#define PATTERN_SIZE 5

/* Granularity represents how long the sequence each thread has to scan is long
*  It affects how much a single thread work
*  Its value also affects the parallelization factor of the GPU
*/ 
#define GRANULARITY_RK 10

/* GPU parameters definition
*/
#define BLOCK_DIMENSION 16

/* Algorithms that can be chosen for test runs
*/
enum algo { NAIVE_RK = 1, RK = 2 };