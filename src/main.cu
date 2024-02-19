#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "../inc/constants.h"
#include "../inc/smatch.h"


/* Prototypes
 */
void parse_args(int argc, char **argv, unsigned char **text, int *text_size, unsigned char ***pattern, int **pattern_size, int *pattern_number, int ***results);
int read_text (char *file_path, unsigned char *storage);
int read_patterns (char *file_path, unsigned char **pattern, int *pattern_size, int *pattern_number);
void cpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results);
void gpu_onept_call (int algorithm, int granularity, int stream_num, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results);
void gpu_multipt_call (int granularity, unsigned char *text, int text_size, unsigned char **pattern, int *pattern_size, int pattern_number, int **results);
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
    if (SCRIPT_MODE == 0) {
        read_program_parameters(&chs_algo, &chs_g, &chs_stream_num, pattern_number);
    }
    else {
        chs_algo = strtol(argv[3], NULL, 10);
        chs_g = strtol(argv[4], NULL, 10);
        chs_stream_num = strtol(argv[5], NULL, 10);
    }

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
        gpu_onept_call(chs_algo, chs_g, chs_stream_num, text, text_size, pattern[0], pattern_size[0], results[0]);
        evaluate_result(results[0], text_size, pattern_size[0]);
    } 
    else {
        gpu_multipt_call(chs_g, text, text_size, pattern, pattern_size, pattern_number, results);

        for (int i = 0; i < pattern_number; i++)
            evaluate_result(results[i], text_size, pattern_size[i]);
    }

    // Release memory
    for (int i = 0; i < pattern_number; i++){
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

    int *loc_patsize, **loc_res;            // Used for better readability of allocations
    unsigned char **loc_pat;                // Used for better readability of allocations
    struct stat text_info, pattern_info;

    // Verify parameters
    if (SCRIPT_MODE == 0 && argc != 3) {
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

    loc_patsize = (int *) malloc((MAX_PATTERN_NUMBER) * sizeof(int));
    loc_pat = (unsigned char **) malloc((MAX_PATTERN_NUMBER) * sizeof(unsigned char *));

    if (text == NULL || loc_pat == NULL || loc_patsize == NULL) {
        fprintf(stderr, "Error during memory allocation!\n");
        exit(EXIT_FAILURE);
    }

    // Read the files after allocation
    if(read_text(argv[1], *text) == 1) {
        fprintf(stderr, "Error during file reading!\n");
        exit(EXIT_FAILURE);
    }

    if(read_patterns(argv[2], loc_pat, loc_patsize, pattern_number) == 1) {
        fprintf(stderr, "Error during file reading!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate results
    loc_res = (int **) malloc((*pattern_number) * sizeof(int *));
    if (loc_res == NULL) {
        fprintf(stderr, "Error during memory allocation!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < (*pattern_number); i++) {
        loc_res[i] = (int *) malloc((*text_size) * sizeof(int));
        
        if (loc_res[i] == NULL) {
            fprintf(stderr, "Error during memory allocation!\n");
            exit(EXIT_FAILURE);
        }
    }

    *pattern = loc_pat;
    *pattern_size = loc_patsize;
    *results = loc_res;
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
        pattern[i] = (unsigned char *) malloc(read * sizeof(unsigned char));
        if(pattern[i] == NULL) {
            fprintf(stderr, "Error during memory allocation!\n");
            exit(EXIT_FAILURE);
        }

        memcpy(pattern[i], pt, read);
        pattern_size[i] = read;       
        i++;
    }
    
    *pattern_number = i;
    for (int j = 0; j < i-1; j++)
        pattern_size[j]--;

    return 0;
}


/* CPU execution of the SMA, for comparison with the GPU one
 */
void cpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results) {

    int *lps, *bshifts, *gshifts;
    double diff;
    struct timeval start, end;

    lps = NULL;
    bshifts = NULL;
    gshifts = NULL;
    gettimeofday(&start, 0);

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

    gettimeofday(&end, 0);
    diff = (1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 1000.0;
    printf("Operations terminated in %f ms.\n", diff);
}


/* Code for the GPU algorithm calls (Single-Pattern)
 *
 * Works only for one pattern only. For multi-pattern search, see the gpu_multipt_call method
 * Designed to allow more flexibility during extensions
 */
void gpu_onept_call (int algorithm, int granularity, int stream_num, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results) {

    int stream_size, text_stream_size;
    float elaboration_time;

    int grid_size_x, grid_size_y, block_size_x, block_size_y;
    int *lps, *gpu_lps, *bshifts, *gpu_bshifts, *gshifts, *gpu_gshifts, *gpu_results;
    float subtext_num, grid_size;
    unsigned char *gpu_text, *gpu_pattern;

    cudaStream_t *stream;
    cudaEvent_t *start, *end;

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
    subtext_num = (float) text_size / (stream_num * granularity);
    subtext_num = ceil(subtext_num);

    grid_size = subtext_num / (block_size_x * block_size_y);
    grid_size_x = (int) ceil(sqrt(grid_size));

    if (ceil(grid_size) == grid_size_x) {
        grid_size_y = 1;
    }
    else {
        grid_size_y = grid_size_x;
    }


    dim3 gridDimension(grid_size_x, grid_size_y);
    dim3 blockDimension(block_size_x, block_size_y);
    printf("Text size: %d bytes\nPattern size: %d bytes\n", text_size, pattern_size);
    printf("Granularity: %d\nSubtext's number: %.0f\n", granularity, subtext_num);
    printf("Stream allocated: %d\n", stream_num);
    printf("Grid (per stream): %dx%d\nBlocks: %dx%d\n\n", grid_size_x, grid_size_y, block_size_x, block_size_y);

    // Streams & Events
    stream = (cudaStream_t *) malloc(stream_num * sizeof(cudaStream_t));
    start = (cudaEvent_t *) malloc(stream_num * sizeof(cudaEvent_t));
    end = (cudaEvent_t *) malloc(stream_num * sizeof(cudaEvent_t));
    for (int i = 0; i < stream_num; i++) {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&(start[i]));
        cudaEventCreate(&(end[i]));
    }

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
                cudaEventRecord(start[i], stream[i]);

                if (algorithm == NAIVE_RK)
                    naive_rk_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, granularity, gpu_results + i * stream_size);
                else
                    rk_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, granularity, gpu_results + i * stream_size);
                
                cudaEventRecord(end[i], stream[i]);
                cudaEventSynchronize(end[i]);
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
                cudaEventRecord(start[i], stream[i]);

                if (algorithm == NAIVE_KMP)
                    naive_kmp_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_lps, granularity, gpu_results + i * stream_size);
                else
                    kmp_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_lps, granularity, gpu_results + i * stream_size);
                    
                cudaEventRecord(end[i], stream[i]);
                cudaEventSynchronize(end[i]);
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
                cudaEventRecord(start[i], stream[i]);

                if (algorithm == NAIVE_BM)
                    naive_boyer_moore_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_bshifts, gpu_gshifts, granularity, gpu_results + i * stream_size);
                else
                    boyer_moore_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_bshifts, gpu_gshifts, granularity, gpu_results + i * stream_size);
                    
                cudaEventRecord(end[i], stream[i]);
                cudaEventSynchronize(end[i]);
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
    for (int i = 0; i < stream_num; i++) {
        cudaEventElapsedTime(&elaboration_time, start[i], end[i]);
        printf("Kernel operations of stream %d terminated in %f ms\n", i, elaboration_time);
    }

    // Streams & Events
    for (int i = 0; i < stream_num; i++) {
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(start[i]);
        cudaEventDestroy(end[i]);
    }

    // Free GPU memory
    cudaFree(gpu_text);
    cudaFree(gpu_pattern);
    cudaFree(gpu_results);
    cudaFree(gpu_lps);
    cudaFree(gpu_bshifts);
    cudaFree(gpu_gshifts);

    free(stream);
    free(start);
    free(end);
}


/* Code for the GPU algorithm calls (Multi-Pattern)
 *
 * Called only when multiple pattern have to be searched. Single pattern search is performed by the gpu_onept_call method.
 * Only RK is supported for simplicity in memory allocation and management
 */
void gpu_multipt_call (int granularity, unsigned char *text, int text_size, unsigned char **pattern, int *pattern_size, int pattern_number, int **results) {

    int grid_size_x, grid_size_y, pt_stream;
    int *gpu_results;
    float grid_size, subtext_num, elaboration_time;
    unsigned char *gpu_text, *gpu_pattern;

    cudaStream_t stream[MAX_MULTIPT_STREAM];
    cudaEvent_t start[MAX_MULTIPT_STREAM], end[MAX_MULTIPT_STREAM];

    // Kernel parameters definition
    printf("Defining grid and block dimensions...\n");
    subtext_num = (float) text_size / granularity;
    subtext_num = ceil(subtext_num);

    grid_size = subtext_num / (BLOCK_DIMENSION * BLOCK_DIMENSION);
    grid_size_x = (int) ceil(sqrt(grid_size));

    if (ceil(grid_size) == grid_size_x) {
        grid_size_y = 1;
    }
    else {
        grid_size_y = grid_size_x;
    }

    dim3 gridDimension(grid_size_x, grid_size_y);
    dim3 blockDimension(BLOCK_DIMENSION, BLOCK_DIMENSION);
    printf("Text size: %d bytes\n", text_size);
    for (int i = 0; i < pattern_number; i++) 
        printf("Pattern #%d size: %d bytes\n", i+1, pattern_size[i]);

    printf("Granularity: %d\nSubtext's number: %.0f\n", granularity, subtext_num);
    printf("Stream allocated: %d\n", MAX_MULTIPT_STREAM);
    printf("Grid (per stream): %dx%d\nBlocks: %dx%d\n\n", grid_size_x, grid_size_y, BLOCK_DIMENSION, BLOCK_DIMENSION);

    // Streams & Events
    for (int i = 0; i < MAX_MULTIPT_STREAM; i++) {
        cudaStreamCreate(&stream[i]);
        cudaEventCreate(&(start[i]));
        cudaEventCreate(&(end[i]));
    }

    // GPU allocations and copy
    cudaMalloc((void **) &gpu_text, text_size * sizeof(unsigned char));
    cudaMemcpy(gpu_text, text, text_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &gpu_pattern, pattern_number * MAX_PATTERN_LENGTH * sizeof(unsigned char));
    cudaMalloc((void **) &gpu_results, pattern_number * text_size * sizeof(int));

    // Kernel launch
    for (int i = 0; i < pattern_number; i++) {
        pt_stream = i % MAX_MULTIPT_STREAM;
        printf("Launching the kernel for pattern %d using stream %d...\n", i+1, pt_stream);
        
        cudaMemcpyAsync(gpu_pattern + i * MAX_PATTERN_LENGTH, pattern[i], pattern_size[i] * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[pt_stream]);
        cudaEventRecord(start[i], stream[i]);

        rk_gpu<<<gridDimension, blockDimension, 0, stream[pt_stream]>>>(gpu_text, text_size, gpu_pattern + i * MAX_PATTERN_LENGTH, pattern_size[i], granularity, gpu_results + i * text_size);
        
        cudaEventRecord(end[i], stream[i]);
        cudaEventSynchronize(end[i]);
        cudaMemcpyAsync(results[i], gpu_results + i * text_size, text_size * sizeof(int), cudaMemcpyDeviceToHost, stream[pt_stream]);
    }
    cudaDeviceSynchronize();

    // Compute time for elaboration
    for (int i = 0; i < pattern_number; i++) {
        cudaEventElapsedTime(&elaboration_time, start[i], end[i]);
        printf("Kernel operations of stream %d terminated in %f ms\n", i, elaboration_time);
    }
         
    // Streams & Events
    for (int i = 0; i < MAX_MULTIPT_STREAM; i++) {
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(start[i]);
        cudaEventDestroy(end[i]);
    }

    // Free GPU memory
    cudaFree(gpu_pattern);
    cudaFree(gpu_results);
    cudaFree(gpu_text);
    cudaFree(gpu_pattern);
    cudaFree(gpu_results);
}


/* Evaluate the results obtained
*/ 
int evaluate_result(int *results, int text_size, int pattern_size) {

    int matches = 0;
    printf("\n");

    for (int i = 0; i < (text_size - pattern_size + 1); i++){
        if (results[i] >= 1){
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

    if (pattern_number == 1) {
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
    } 
    else {
        *chs_algo = RK;
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