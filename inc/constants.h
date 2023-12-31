/* Location where the pattern to search is defined/stored
*  1 - Constant memory; 2 - Read from stdin or defined in main function
*/
#define PATTERN_LOCATION 2

/* Pattern length
*  Used when this is statically defined
*/
#define PATTERN_SIZE 5

/* Used to define the granularity required from the RK-algorithm
*  g = TEXT_SIZE / SUBTEXTS_NUM
*  
*  The lower the value of g, the more threads are required to analyze everything
*  The higher the value of g, the higher will be the time required by each thread to complete the search
*/
#define SUBTEXTS_NUM 5000

/* GPU parameters definition
*/
#define BLOCK_DIMENSION 16

/* Algorithms that can be chosen for test runs
*/
enum algo { NAIVE_RK = 1, RK = 2 };