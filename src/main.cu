#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

#include "../inc/constants.h"
#include "../inc/smatch.h"

// TEMP
const int ALGO = NAIVE_BM;


/* Prototypes
*/
int read_text (char *file_path, unsigned char *storage);
void cpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results);
void gpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results);
int evaluate_result(int *results, int text_size, int pattern_size);
void print_gpu_properties();

int main (int argc, char *argv[]) {

    int text_size, pattern_size;
    int *results;
    unsigned char *text;
    unsigned char *pattern;
    unsigned char stat_pattern[] = "ipsum";

    struct stat text_info, pattern_info;

    // Verify parameters
    if (argc < 2 && argc > 3) {
        fprintf(stderr, "Call to the program performed incorrectly!\n");
        fprintf(stderr, "Send only two arguments at maximum: the path for the text to be searched on and the path for the pattern to use\n");
        exit(EXIT_FAILURE);
    }

    if (stat(argv[1], &text_info) == -1) {
        fprintf(stderr, "Indicated path for text might not be correct\n");
        exit(EXIT_FAILURE);
    }

    text_size = text_info.st_size;
    text = (unsigned char*) malloc(text_size * sizeof(unsigned char));

    if (argc == 3) {
        if (stat(argv[2], &pattern_info) == -1) {
            fprintf(stderr, "Indicated path for pattern might not be correct\n");
            exit(EXIT_FAILURE);
        }
    
        pattern_size = pattern_info.st_size;
        pattern = (unsigned char *) malloc(pattern_size * sizeof(unsigned char));    
    }
    else {
        pattern_size = strlen((char *) stat_pattern);
        pattern = stat_pattern;
    }

    results = (int*) malloc(text_size * sizeof(int));

    if (text == NULL || pattern == NULL || results == NULL) {
        fprintf(stderr, "Error during memory allocation!\n");
        exit(EXIT_FAILURE);
    }
    
    if(read_text(argv[1], text) == 1) {
        fprintf(stderr, "Error during file reading!\n");
        exit(EXIT_FAILURE);
    }

    if(argc == 3 && read_text(argv[2], pattern) == 1) {
        fprintf(stderr, "Error during file reading!\n");
        exit(EXIT_FAILURE);
    }

    print_gpu_properties();
    printf("Launching the algorithm on the host device (CPU)...\n");
    cpu_call(ALGO, text, text_size, pattern, pattern_size, results);
    evaluate_result(results, text_size, pattern_size);

    printf("Launching the algorithm on the device (GPU)...\n");
    //gpu_call(ALGO, text, text_size, pattern, pattern_size, results);
    evaluate_result(results, text_size, pattern_size);

    // Release memory
    free(text);
    return 0;
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
    printf("Operations terminated in %f seconds.\n", diff);

}


/* Code for the GPU algorithm calls

*  Some parts, especially inside the switch statement, may seem redundant but it is designed 
*  in this way to allow more flexibility in case some calls require it
*/
void gpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results) {

    int stream_size, text_stream_size;
    float elaboration_time[MAX_NUM_STREAM];

    int grid_size_x, grid_size_y, block_size_x, block_size_y, subtext_num;
    int *lps, *gpu_lps, *gpu_results;
    unsigned char *gpu_text, *gpu_pattern;

    cudaStream_t stream[MAX_NUM_STREAM];
    cudaEvent_t start, end;

    // Kernel parameters definition
    printf("Defining grid and block dimensions...\n");
    block_size_x = BLOCK_DIMENSION;
    block_size_y = BLOCK_DIMENSION;

    switch (algorithm) {
        case NAIVE_RK:
        case RK:
            /* Used to define the granularity required from the RK-algorithm
            *  g = TEXT_SIZE / SUBTEXTS_NUM
            *  
            *  The lower the value of g, the more threads are required to analyze everything
            *  The higher the value of g, the higher will be the time required by each thread to complete the search
            */
            subtext_num = ceil(text_size / (MAX_NUM_STREAM * GRANULARITY_RK)) + 1;
            break;

        case NAIVE_KMP:
        case KMP:
            /* Used to define the granularity required from the RK-algorithm
            *  g = TEXT_SIZE / SUBTEXTS_NUM
            *  
            *  The lower the value of g, the more threads are required to analyze everything
            *  The higher the value of g, the higher will be the time required by each thread to complete the search
            */
            subtext_num = ceil(text_size / (MAX_NUM_STREAM * GRANULARITY_KMP)) + 1;
            break;

        default:
            fprintf(stderr, "Error: chosen algorithm not supported!\n");
            exit(EXIT_FAILURE);
            break;
    }

    grid_size_x = ceil(sqrt(subtext_num / (block_size_x * block_size_y))) + 1;
    grid_size_y = grid_size_x;

    dim3 gridDimension(grid_size_x, grid_size_y);
    dim3 blockDimension(block_size_x, block_size_y);
    printf("Text size: %d bytes\nPattern size: %d bytes\n", text_size, pattern_size);
    printf("Stream allocated: %d\n", MAX_NUM_STREAM);
    printf("Grid (per stream): %dx%d\nBlocks: %dx%d\n\n", grid_size_x, grid_size_y, block_size_x, block_size_y);

    // Streams
    for (int i = 0; i < MAX_NUM_STREAM; i++)
        cudaStreamCreate(&stream[i]);

    // Events
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // GPU allocations and copy
    cudaMalloc((void **) &gpu_text, text_size * sizeof(unsigned char));
    cudaMalloc((void **) &gpu_pattern, pattern_size * sizeof(unsigned char));
    cudaMalloc((void **) &gpu_results, text_size * sizeof(int));
    cudaMemcpy(gpu_pattern, pattern, pattern_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    stream_size = text_size / MAX_NUM_STREAM;
    text_stream_size = stream_size + pattern_size;
                
    switch (algorithm) {
        case NAIVE_RK:        
        case RK:
            for (int i = 0; i < MAX_NUM_STREAM; i++) {
                printf("Launching the kernel using stream %d...\n", i);

                /* On the last round, make some adjustment for incorrect memory management due
                *  to rounding errors
                */
                if (i == MAX_NUM_STREAM - 1)
                    text_stream_size = text_size - i * stream_size;

                cudaMemcpyAsync(gpu_text + i * stream_size, text + i * stream_size, text_stream_size * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[i]);

                if (algorithm == NAIVE_RK)
                    naive_rk_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, GRANULARITY_RK, gpu_results + i * stream_size);
                else
                    rk_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, GRANULARITY_RK, gpu_results + i * stream_size);
                
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

            for (int i = 0; i < MAX_NUM_STREAM; i++) {             
                printf("Launching the kernel using stream %d...\n", i);

                /* On the last round, make some adjustment for incorrect memory management due
                *  to rounding errors
                */
                if (i == MAX_NUM_STREAM - 1)
                    text_stream_size = text_size - i * stream_size;

                cudaMemcpyAsync(gpu_text + i * stream_size, text + i * stream_size, text_stream_size * sizeof(unsigned char), cudaMemcpyHostToDevice, stream[i]);

                if (algorithm == NAIVE_KMP)
                    naive_kmp_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_stream_size, gpu_pattern, pattern_size, gpu_lps, GRANULARITY_KMP, gpu_results + i * stream_size);
                else
                    kmp_gpu<<<gridDimension, blockDimension, 0, stream[i]>>>(gpu_text + i * stream_size, text_size, gpu_pattern, pattern_size, gpu_lps, GRANULARITY_KMP, gpu_results + i * stream_size);
                    
                cudaMemcpyAsync(results + i * stream_size, gpu_results + i * stream_size, text_stream_size * sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
            }

            cudaDeviceSynchronize();
            free(lps);
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
    for (int i = 0; i < MAX_NUM_STREAM; i++)
        cudaStreamDestroy(stream[i]);

    // Events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Free GPU memory
    cudaFree(gpu_text);
    cudaFree(gpu_pattern);
    cudaFree(gpu_results);
    cudaFree(gpu_lps);
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