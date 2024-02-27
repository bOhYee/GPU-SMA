# GPU-SMA
Project realized for the GPU Programming course at Polytechnic of Turin. 
The goal was to implement three state-of-the-art String Matching Algorithms (SMA) as kernels for GPU execution and analyze their performance with respect to the serial counterparts. Due to the scope of the project, the kernels were built only for Nvidia GPUs, using the CUDA library.
The three selected algorithms are the **Rabin-Karp**, the **Knuth-Morris-Pratt** and the **Boyer-Moore**. Due to their strong sequential nature, the kernel's structure couldn't be optimized for parallel execution, thus, the followed approach was that of a "Divide and Conquer" strategy. The text is divided among the many threads to be scanned, reducing elaboration times.

The program requires two arguments: the path to the text's file and the path to the pattern's file. The latter can contain different strings on each line, which will be searched separately by the program on the provided text. **Up to 8 searches are supported**. At start-up, it will ask the user to define some parameters before launching the scan operation:
- the algorithm to use for the search;
- the number of CUDA streams to use (only for optimized algorithms);
- the length of the substrings searched by each thread (**granularity** value).

When more patterns are searched, only the Rabin-Karp algorithm will be used and the number of streams is set to the number of patterns provided.

A more thorough description of the technologies employed, the algorithms' strategies, the implementation and results' evaluation can be found on the [project report](docs/GPU-Report.pdf).

## Install
To compile the program the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is required. In case you have it already installed and configured, you can compile the program by using the Makefile that can be found on the repository:

```
mkdir obj
make
```