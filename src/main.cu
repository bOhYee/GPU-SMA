#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

#include "../inc/constants.h"
#include "../inc/smatch.h"

// TEMP
const int ALGO = KMP;


/* Prototypes
 */
void parse_args(int argc, char **argv, unsigned char **text, int *text_size, unsigned char ***pattern, int **pattern_size, int *pattern_number, int ***results);
int read_text (char *file_path, unsigned char *storage);
int read_patterns (char *file_path, unsigned char **pattern, int *pattern_size, int *pattern_number);
void cpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results);
void gpu_call (int algorithm, int granularity, int stream_num, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results);
int evaluate_result(int *results, int text_size, int pattern_size);
void print_gpu_properties();
void read_program_parameters(int *chs_algo, int *chs_g, int *chs_stream_num, int pattern_number);


int main (int argc, char *argv[]) {

    int chs_algo, chs_stream_num, chs_g;
    int text_size, pattern_number, *pattern_size;
    int **results;
    unsigned char *text, **pattern;

    // print_gpu_properties();
    parse_args(argc, argv, &text, &text_size, &pattern, &pattern_size, &pattern_number, &results);
    read_program_parameters(&chs_algo, &chs_g, &chs_stream_num, pattern_number);

    // Init results 
    for (int i = 0; i < pattern_number; i++) {
        for (int j = 0; j < text_size; j++) 
            results[i][j] = 0;
    }

    printf("###################################################\n");
    printf("Launching the algorithm on the host device (CPU)...\n");
    printf("###################################################\n");
    for (int i = 0; i < pattern_number; i++)
        cpu_call(chs_algo, text, text_size, pattern[i], pattern_size[i], results[i]);

    for (int i = 0; i < pattern_number; i++){
        printf("Results for pattern %d:", i+1);
        evaluate_result(results[i], text_size, pattern_size[i]);
    }

    printf("##############################################\n");
    printf("Launching the algorithm on the device (GPU)...\n");
    printf("##############################################\n");
    if (pattern_number == 1) {
        gpu_call(chs_algo, chs_g, chs_stream_num, text, text_size, pattern[0], pattern_size[0], results[0]);
        evaluate_result(results[0], text_size, pattern_size[0]);
    }

    // Release memory
    for (int i = 0; i < MAX_PATTERN_NUMBER; i++){
        free(pattern[i]);
        free(results[i]);
    }

    free(text);
    free(pattern_size);
    free(pattern);
    free(results);
    return 0;
}


/* Parse the arguments provided by the command line interface
 */
void parse_args(int argc, char **argv, unsigned char **text, int *text_size, unsigned char ***pattern, int **pattern_size, int *pattern_number, int ***results) {

    struct stat text_info, pattern_info;

    // Verify parameters
    if (argc != 3) {
        fprintf(stderr, "Call to the program performed incorrectly!\n");
        fprintf(stderr, "Send only two arguments: the path for the text to be searched on and the path for the pattern to use\n");
        exit(EXIT_FAILURE);
    }

    // TEXT to be searched
    if (stat(argv[1], &text_info) == -1) {
        fprintf(stderr, "Indicated path for text might not be correct\n");
        exit(EXIT_FAILURE);
    }

    // PATTERN to be searched
    if (stat(argv[2], &pattern_info) == -1) {
        fprintf(stderr, "Indicated path for pattern might not be correct\n");
        exit(EXIT_FAILURE);
    }

    *text_size = (int) text_info.st_size;
    *text = (unsigned char*) malloc((*text_size) * sizeof(unsigned char));
    *results = (int**) malloc(MAX_PATTERN_NUMBER * sizeof(int*));
    *pattern_size = (int *) malloc((MAX_PATTERN_NUMBER) * sizeof(int));
    *pattern = (unsigned char **) malloc((MAX_PATTERN_NUMBER) * sizeof(unsigned char*));

    if (text == NULL || pattern == NULL || pattern_size == NULL || results == NULL) {
        fprintf(stderr, "Error during memory allocation!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < MAX_PATTERN_NUMBER; i++) {
        (*results)[i] = (int*) malloc((*text_size) * sizeof(int));
        
        if ((*results)[i] == NULL) {
            fprintf(stderr, "Error during memory allocation!\n");
            exit(EXIT_FAILURE);
        }
    }

    // Read the files after allocation
    if(read_text(argv[1], *text) == 1) {
        fprintf(stderr, "Error during file reading!\n");
        exit(EXIT_FAILURE);
    }

    if(read_patterns(argv[2], *pattern, *pattern_size, pattern_number) == 1) {
        fprintf(stderr, "Error during file reading!\n");
        exit(EXIT_FAILURE);
    }
}


/* Copy the text from the file inside the memory
 */
int read_text (char *file_path, unsigned char *storage) {

    int i;
    FILE *in_file;

    in_file = fopen(file_path, "r");
    if (in_file == NULL) 
        return 1;

    i = 0;
    while (fscanf(in_file, "%c", &storage[i++]) > 0);

    return 0;
}


/* Copy the patterns from the file to the memory
 */
int read_patterns (char *file_path, unsigned char **pattern, int *pattern_size, int *pattern_number) {

    char *pt;
    int i, read, pt_len;
    FILE *in_file;

    in_file = fopen(file_path, "r");
    if (in_file == NULL)
        return 1;

    i = 0;
    pt = NULL;
    while ((read = getline(&pt, (size_t *) &pt_len, in_file)) != -1 && i < MAX_PATTERN_NUMBER) {
        pattern[i] = (unsigned char *) malloc((read-1) * sizeof(unsigned char));
        if(pattern[i] == NULL) {
            fprintf(stderr, "Error during memory allocation!\n");
            exit(EXIT_FAILURE);
        }

        memcpy(pattern[i], pt, read-1);
        pattern_size[i] = read - 1;       
        i++;
    }
    
    *pattern_number = i;
    return 0;
}


/* CPU execution of the SMA, for comparison with the GPU one
 */
void cpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results) {

    int *lps, *bshifts, *gshifts;
    double diff;
    time_t start;

    lps = NULL;
    bshifts = NULL;
    gshifts = NULL;
    start = time(NULL);

    switch (algorithm) {
        case NAIVE_RK:
        case RK:
            rk_cpu(text, text_size, pattern, pattern_size, results);
            break;
        
        case NAIVE_KMP:
        case KMP:
            lps = (int *) malloc(pattern_size * sizeof(int));
            if (lps == NULL) {
                fprintf(stderr, "Error during memory allocation of LPS vector!\n");
                exit(EXIT_FAILURE);
            }
            
            kmp_cpu(text, text_size, pattern, pattern_size, lps, results);
            free(lps);
            break;

        case NAIVE_BM:
        case BM:
            bshifts = (int *) malloc(ALPHABET_SIZE * sizeof(int));
            gshifts = (int *) malloc(pattern_size * sizeof(int));
            if (bshifts == NULL || gshifts == NULL) {
                fprintf(stderr, "Error during memory allocation of BadCharacterRule or GoodSuffixRule vectors!\n");
                exit(EXIT_FAILURE);
            }

            boyer_moore_cpu(text, text_size, pattern, pattern_size, bshifts, gshifts, results);
            free(bshifts);
            free(gshifts);
            break;

        default:
            fprintf(stderr, "Error: chosen algorithm not supported!\n");
            exit(EXIT_FAILURE);
            break;
    }

    diff = difftime(time(NULL), start);
   // printf("Operations terminated in %f seconds.\n", diff);
}


/* Code for the GPU algorithm calls
 *
 * Some parts, especially inside the switch statement, may seem redundant but it is designed 
 * in this way to allow more flexibility in case some calls require it
 */
void gpu_call (int algorithm, int granularity, int stream_num, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results) {

    int stream_size, text_stream_size;
    float elaboration_time[MAX_NUM_STREAM];

    int grid_size_x, grid_size_y, block_size_x, block_size_y, subtext_num;
    int *lps, *gpu_lps, *bshifts, *gpu_bshifts, *gshifts, *gpu_gshifts, *gpu_results;
    unsigned char *gpu_text, *gpu_pattern;

    cudaStream_t *stream;
    cudaEvent_t start, end;

    // Kernel parameters definition
    printf("Defining grid and block dimensions...\n");
    block_size_x = BLOCK_DIMENSION;
    block_size_y = BLOCK_DIMENSION;

    /* Used to define the granularity required from the RK-algorithm
    *  g = TEXT_SIZE / SUBTEXTS_NUM
    *  
    *  The lower the value of g, the more threads are required to analyze everything
    *  The higher the value of g, the higher will be the time required by each thread to complete the search
    */
    subtext_num = ceil(text_size / (stream_num * granularity)) + 1;
    grid_size_x = ceil(sqrt(subtext_num / (block_size_x * block_size_y))) + 1;
    grid_size_y = grid_size_x;

    dim3 gridDimension(grid_size_x, grid_size_y);
    dim3 blockDimension(block_size_x, block_size_y);
    printf("Text size: %d bytes\nPattern size: %d bytes\n", text_size, pattern_size);
    printf("Stream allocated: %d\n", stream_num);
    printf("Grid (per stream): %dx%d\nBlocks: %dx%d\n\n", grid_size_x, grid_size_y, block_size_x, block_size_y);

    // Streams
    stream = (cudaStream_t *) malloc(stream_num * sizeof(cudaStream_t));
    for (int i = 0; i < stream_num; i++)
        cudaStreamCreate(&stream[i]);

    // Events
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // GPU allocations and copy
    cudaMalloc((void **) &gpu_text, text_size * sizeof(unsigned char));
    cudaMalloc((void **) &gpu_pattern, pattern_size * sizeof(unsigned char));
    cudaMalloc((void **) &gpu_results, text_size * sizeof(int));
    cudaMemcpy(gpu_pattern, pattern, pattern_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    stream_size = text_size / stream_num;
    text_stream_size = stream_size + pattern_size;
                
    switch (algorithm) {
        case NAIVE_RK:        
        case RK:
            for (int i = 0; i < stream_num; i++) {
                printf("Launching the kernel using stream %d...\n", i);

                /* On the last round, make some adjustment for incorrect memory management due
                *  to rounding errors
                */
                if (i == stream_num - 1)
                    text_stream_size = text_size - i * stream_size;

                cudaMemcpyAsync(gpu_text + i * stream_size, text + i * stream_size, text_stream_size * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[i]);

                if (algorithm == NAIVE_RK)
                    naive_rk_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, granularity, gpu_results + i * stream_size);
                else
                    rk_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, granularity, gpu_results + i * stream_size);
                
                cudaMemcpyAsync(results + i * stream_size, gpu_results + i * stream_size, text_stream_size * sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
            }
            
            cudaDeviceSynchronize();
            break;

        case NAIVE_KMP:
        case KMP:
            lps = (int *) malloc(pattern_size * sizeof(int));
            if (lps == NULL) {
                fprintf(stderr, "Error during memory allocation of LPS vector in device code!\n");
                exit(EXIT_FAILURE);
            }

            compute_lps(pattern, pattern_size, lps);
            cudaMalloc((void **) &gpu_lps, pattern_size * sizeof(unsigned int));
            cudaMemcpy(gpu_lps, lps, pattern_size * sizeof(int), cudaMemcpyHostToDevice);

            for (int i = 0; i < stream_num; i++) {             
                printf("Launching the kernel using stream %d...\n", i);

                /* On the last round, make some adjustment for incorrect memory management due
                *  to rounding errors
                */
                if (i == stream_num - 1)
                    text_stream_size = text_size - i * stream_size;

                cudaMemcpyAsync(gpu_text + i * stream_size, text + i * stream_size, text_stream_size * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[i]);

                if (algorithm == NAIVE_KMP)
                    naive_kmp_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_lps, granularity, gpu_results + i * stream_size);
                else
                    kmp_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_lps, granularity, gpu_results + i * stream_size);
                    
                cudaMemcpyAsync(results + i * stream_size, gpu_results + i * stream_size, text_stream_size * sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
            }

            cudaDeviceSynchronize();
            free(lps);
            break;
        
        case NAIVE_BM:
        case BM:
            bshifts = (int *) malloc(ALPHABET_SIZE * sizeof(int));
            gshifts = (int *) malloc(pattern_size * sizeof(int));
            if (bshifts == NULL || gshifts == NULL) {
                fprintf(stderr, "Error during memory allocation of bad rule and good rule tables in device code!\n");
                exit(EXIT_FAILURE);
            }

            /* Preprocessing not possibile on the device due to the high amount of possible bank conficts, even if divide and conquer approach used
             *  Better to preprocess the patter using the CPU and leave only the search to the device
             */ 
            bad_char_rule(bshifts, pattern, pattern_size);
            good_suffix_rule(gshifts, pattern, pattern_size);

            cudaMalloc((void **) &gpu_bshifts, ALPHABET_SIZE * sizeof(unsigned int));
            cudaMalloc((void **) &gpu_gshifts, pattern_size * sizeof(unsigned int));
            cudaMemcpy(gpu_bshifts, bshifts, ALPHABET_SIZE * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_gshifts, gshifts, pattern_size * sizeof(int), cudaMemcpyHostToDevice);    

            for (int i = 0; i < stream_num; i++) {             
                printf("Launching the kernel using stream %d...\n", i);

                /* On the last round, make some adjustment for incorrect memory management due
                 *  to rounding errors
                 */
                if (i == stream_num - 1)
                    text_stream_size = text_size - i * stream_size;

                cudaMemcpyAsync(gpu_text + i * stream_size, text + i * stream_size, text_stream_size * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[i]);

                if (algorithm == NAIVE_BM)
                    naive_boyer_moore_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_bshifts, gpu_gshifts, granularity, gpu_results + i * stream_size);
                else
                    boyer_moore_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_bshifts, gpu_gshifts, granularity, gpu_results + i * stream_size);
                    
                cudaMemcpyAsync(results + i * stream_size, gpu_results + i * stream_size, text_stream_size * sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
            }

            cudaDeviceSynchronize();
            free(bshifts);
            free(gshifts);
            break;

        default:
            fprintf(stderr, "Error: chosen algorithm not supported!\n");
            exit(EXIT_FAILURE);
            break;
    }

    // Compute time for elaboration
    //cudaEventElapsedTime(&elaboration_time, start, end);
    //elaboration_time /= 1000;
    //printf("Kernel operations terminated in %f seconds\n", elaboration_time);

    // Streams
    for (int i = 0; i < stream_num; i++)
        cudaStreamDestroy(stream[i]);

    // Events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Free GPU memory
    cudaFree(gpu_text);
    cudaFree(gpu_pattern);
    cudaFree(gpu_results);
    cudaFree(gpu_lps);
    cudaFree(gpu_bshifts);
    cudaFree(gpu_gshifts);

    free(stream);
}


/* Evaluate the results obtained
*/ 
int evaluate_result(int *results, int text_size, int pattern_size) {

    int matches = 0;
    printf("\n");

    for (int i = 0; i < (text_size - pattern_size + 1); i++){
        if (results[i] == 1){
            matches++;
            printf("Match found at index: %d\n", i+1);
        }
    }

    printf("Total matches: %d\n\n", matches);

    // Reset the results array after scanning it
    for (int i = 0; i < (text_size - pattern_size + 1); i++)
        results[i] = 0;
        
    return matches;
}


/* Prints informations about the CUDA device to use
*/
void print_gpu_properties() {

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("Printing CUDA device informations...\n");
    printf("Device name:\t\t\t\t\t\t%s\n", p.name);
    printf("Amount of Global memory (bytes):\t\t\t%lu\n", p.totalGlobalMem);
    printf("Maximum amount of Shared memory per block (bytes):\t%lu\n", p.sharedMemPerBlock);
    printf("Amount of Constant memory (bytes):\t\t\t%lu\n", p.totalConstMem);
    printf("Number of SM:\t\t\t\t\t\t%d\n", p.multiProcessorCount);
    printf("Concurrent kernels supported:\t\t\t\t%d\n", p.concurrentKernels);
    printf("\n");

}


/* Recover information about the program execution
 */
void read_program_parameters(int *chs_algo, int *chs_g, int *chs_stream_num, int pattern_number) {

    char line[10];

    printf("Choose the algorithm to execute: \n");
    printf("1) Rabin Karp (naive implementation);\n");
    printf("2) Rabin Karp (optimized);\n");
    printf("3) KMP (naive implementation);\n");
    printf("4) KMP (optimized);\n");
    printf("5) Boyer-Moore (naive implementation);\n");
    printf("6) Boyer-Moore (optimized);\n: ");
    fgets(line, 10, stdin);
    *chs_algo = strtol(line, NULL, 10);

    if (*chs_algo <= 0 || *chs_algo >= 7) {
        fprintf(stderr, "Inserted value is incorrect!\nValue must be a number between 1 and 6. Aborting operations...\n");
        exit(EXIT_FAILURE);
    }

    if ((*chs_algo % 2 == 0) && (pattern_number == 1)) {
        printf("\nChoose how many streams to launch on the GPU: (recommended = 8)\n: ");
        fgets(line, 10, stdin);
        *chs_stream_num = strtol(line, NULL, 10);

        if (*chs_stream_num == 0)
            *chs_stream_num = MAX_NUM_STREAM;
    }
    else {
        *chs_stream_num = 1;
    }

    printf("\nSelect the subtexts' length to use: \n: ");
    fgets(line, 10, stdin);
    *chs_g = strtol(line, NULL, 10);

    if (*chs_g == 0) {
        switch (*chs_algo) {
            case NAIVE_RK:
            case RK:
                *chs_g = GRANULARITY_RK;
                break;

            case NAIVE_KMP:
            case KMP:
                *chs_g = GRANULARITY_KMP;
                break;

            case NAIVE_BM:
            case BM:
            default:
                *chs_g = GRANULARITY_BM;
                break;
        }
    }

    printf("\n");
}