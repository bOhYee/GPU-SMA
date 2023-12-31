#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

#include "../inc/constants.h"
#include "../inc/smatch.h"

/* Prototypes
*/
int read_text (char *file_path, unsigned char *storage);
void cpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results);
void gpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results);

int main (int argc, char *argv[]) {

    int text_size, pattern_size, matches;
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

    results = (int*) malloc((text_size - pattern_size) * sizeof(int));

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

    printf("%d %d", text_size, pattern_size);
    //cpu_call(1, text, text_size, pattern, pattern_size, results);
    gpu_call(1, text, text_size, pattern, pattern_size, results);

    // Results
    matches = 0;
    for (int i = 0; i < (text_size - pattern_size); i++){
        if (results[i] == 1){
            matches++;
            printf("Match found at index: %d\n", i+1);
        }
    }
    printf("Total matches: %d\n", matches);

    // Release memory
    free(text);
    return 0;
}

// Copy the text from the file inside the memory
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

    double diff;
    time_t start, end;

    printf("Launching the algorithm on the host device (CPU)...\n");
    time(&start);

    switch (algorithm) {
        case NAIVE_RK:
        default:
            rk_cpu(text, text_size, pattern, pattern_size, results);
            break;
    }

    time(&end);
    diff = difftime(end, start);
    printf("Operations terminated in %lf seconds.\n", diff);

}

void gpu_call (int algorithm, unsigned char *text, int text_size, unsigned char *pattern, int pattern_size, int *results) {

    int grid_size_x, grid_size_y, block_size_x, block_size_y;
    int *gpu_results;
    float elaboration_time;
    unsigned char *gpu_text, *gpu_pattern;
    cudaEvent_t start, end;

    // Kernel parameters definition
    printf("Defining grid and block dimensions...\n");
    switch (algorithm) {
        case NAIVE_RK:
        default:
            block_size_x = BLOCK_DIMENSION;
            block_size_y = BLOCK_DIMENSION;

            grid_size_x = ceil(sqrt(text_size / (block_size_x * block_size_y))) + 1;
            grid_size_y = grid_size_x;
            break;
    }

    dim3 gridDimension(grid_size_x, grid_size_y);
    dim3 blockDimension(block_size_x, block_size_y);
    printf("Grid: %dx%d\nBlocks: %dx%d\n", grid_size_x, grid_size_y, block_size_x, block_size_y);

    // Events
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // GPU allocations and copy
    cudaMalloc((void **) &gpu_text, text_size * sizeof(unsigned char));
    cudaMalloc((void **) &gpu_pattern, pattern_size * sizeof(unsigned char));
    cudaMemcpy(gpu_text, text, text_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_pattern, pattern, pattern_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    printf("Launching the kernel...\n");
    cudaEventRecord(start);
    switch (algorithm) {
        case NAIVE_RK:
        default:
            cudaMalloc((void **) &gpu_results, (text_size-pattern_size) * sizeof(int));
            naive_rk_gpu<<<gridDimension, blockDimension>>>(gpu_text, text_size, gpu_pattern, pattern_size, ceil(text_size / SUBTEXTS_NUM), gpu_results);
            cudaMemcpy(results, gpu_results, (text_size-pattern_size) * sizeof(int), cudaMemcpyDeviceToHost);
            break;
    }
    cudaEventRecord(end);

    // Compute time for elaboration
    cudaEventElapsedTime(&elaboration_time, start, end);
    //elaboration_time /= 1000;
    printf("Kernel operations terminated in %f seconds\n", elaboration_time);
}