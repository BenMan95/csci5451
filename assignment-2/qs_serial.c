#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Gives us high-resolution timers. */
#undef _POSIX_C_SOURCE // This prevents the compiler displaying a warning
#define _POSIX_C_SOURCE 199309L
#include <time.h>

/**
 * @brief Return the number of seconds since an unspecified time (e.g., Unix
 *        epoch). This is accomplished with a high-resolution monotonic timer,
 *        suitable for performance timing.
 *
 * @return The number of seconds.
 */
static inline double monotonic_seconds()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
} 

/**
 * @brief Output the seconds elapsed while sorting. This excludes input and
 *         output time. This should be wallclock time, not CPU time.
 *
 * @param seconds Seconds spent sorting.
 */
static void print_time(double const seconds)
{
    printf("Sort Time: %0.04fs\n", seconds);
}

/**
 * @brief Write an array of integers to a file.
 *
 * @param filename The name of the file to write to.
 * @param numbers The array of numbers.
 * @param nnumbers How many numbers to write.
 */
static void print_numbers(
        char const * const filename,
        uint32_t const * const numbers,
        uint32_t const nnumbers)
{
    FILE * fout;

    /* open file */
    if((fout = fopen(filename, "w")) == NULL) {
        fprintf(stderr, "error opening '%s'\n", filename);
        abort();
    }

    /* write the header */
    fprintf(fout, "%d\n", nnumbers);

    /* write numbers to fout */
    for(uint32_t i = 0; i < nnumbers; ++i) {
        fprintf(fout, "%d\n", numbers[i]);
    }

    fclose(fout);
}

/**
 * @brief Partitions an array in place
 *
 * @param data The array to partition
 * @param size The size of the array
 *
 * @return Returns the index of the pivot element
 */
int partition_inplace(int *data, int size)
{
    int pivot = data[0];
    int i = 0;
    int j = size;

    while (1) {
        // Find elements to swap
        while (data[++i] < pivot);
        while (data[--j] > pivot);

        // Exit once indices meet
        if (i >= j) {
            data[0] = data[j];
            data[j] = pivot;
            return j;
        }

        // Swap elements
        int temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

/**
 * @brief Perform a quicksort in place
 *
 * @param data The array to sort
 * @param size The size of the array
 */
void quicksort_inplace(int *data, int size)
{
    if (size <= 1) return;
    int p = partition_inplace(data, size);
    quicksort_inplace(data, p);
    quicksort_inplace(data+p+1, size-p-1);
}

/**
 * @brief Gets the nth smallest element of an array
 *
 * @param data The array to read from
 * @param size The size of the array
 * @param n Which element to get
 */
int quickselect(int *data, int size, int n)
{
    int p = partition_inplace(data, size);
    if (n < p) return quickselect(data, p, n);
    if (n > p) return quickselect(data+p+1, size-p-1, n-p-1);
    return data[n];
}

/**
 * @brief Partitions an array and saves the positions of those partitions. Not in-place
 *
 * @param data The array to partition
 * @param size The size of the array
 * @param p1 Where to write the end index of the lower partition (non-inclusive)
 * @param p2 Where to write the start index of the upper partition (inclusive)
 */
void partition(int *data, int size, int *p1, int *p2)
{
    int *buf = malloc(size * sizeof(int));

    int pivot = data[size / 2];
    int j = 0;
    int k = size;
    for (int i = 0; i < size; i++) {
        int num = data[i];
        if (num < pivot) {
            buf[j++] = num;
        }
        if (num > pivot) {
            buf[--k] = num;
        }
    }

    for (int i = j; i < k; i++) {
        buf[i] = pivot;
    }

    for (int i = 0; i < size; i++) {
        data[i] = buf[i];
    }

    *p1 = j;
    *p2 = k;
}

/**
 * @brief Sorts an array using quicksort
 *
 * @param data The array to sort
 * @param size The size of the array to sort
 */
void quicksort(int *data, int size)
{
    if (size <= 1)
        return;

    int p1, p2;
    partition(data, size, &p1, &p2);

    quicksort(data, p1);
    quicksort(data+p2, size-p2);
}

int main(int argc, char** argv)
{
    int *data;
    int datasize;

    double start_time, end_time;

    // Exit if wrong number of arguments
    if (argc != 3) {
        printf("Usage: %s <size> <outfile>\n", argv[0]);
        return 0;
    }

    // Read size and exit if invalid
    datasize = atoi(argv[1]);
    if (datasize == 0) {
        printf("Usage: %s <size> <outfile>\n", argv[0]);
        return 0;
    }

    // Seed random number generation
    srand(0);

    // Generate data
    data = malloc(datasize * sizeof(int));
    for (int i = 0; i < datasize; i++) {
        data[i] = rand() % datasize;
    }

    start_time = monotonic_seconds();
    // BEGIN TIMING ------------------------------------------------------------

    quicksort_inplace(data, datasize);

    // END TIMING --------------------------------------------------------------
    end_time = monotonic_seconds();

    print_time(end_time - start_time);
    print_numbers(argv[2], (uint32_t*) data, datasize);

    free(data);
    return 0;
}

