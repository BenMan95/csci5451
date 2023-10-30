#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

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
 * @brief Partitions an array. In-place and serial
 *
 * @param data The array to partition
 * @param size The size of the array
 *
 * @return Returns the index of the pivot element
 */
int partition_serial(int *data, int size)
{
    // Lomuto's partitioning scheme
    int pivot = data[size-1];
    int i = 0;
    for (int j = 0; j < size; j++) {
        if (data[j] < pivot) {
            int temp;
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;

            i++;
        }
    }
    data[size-1] = data[i];
    data[i] = pivot;
    return i;
}

/**
 * @brief Sorts a list with quicksort. In-place and serial
 *
 * @param data The array to sort
 * @param size The size of the array
 */
void quicksort_serial(int *data, int size)
{
    if (size <= 1) return;
    int p = partition_serial(data, size);
    quicksort_serial(data, p);
    quicksort_serial(data+p+1, size-p-1);
}

/**
 * @brief Returns the nth smallest element of an array
 *
 * @param data The array to read from
 * @param size The size of the array
 * @param n Which element to get
 */
int quickselect(int *data, int size, int n)
{
    int p = partition_serial(data, size);
    if (n < p) return quickselect(data, p, n);
    if (n > p) return quickselect(data+p+1, size-p-1, n-p-1);
    return data[n];
}

/**
 * @brief Partitions an array in parallel and saves their positions. After this
 *        operation, each process in the group will contain the partitioned array
 *
 * @param data The array to partition
 * @param datasize The size of the array to partition
 * @param p1 Where to write the end index of the lower partition (non-inclusive)
 * @param p2 Where to write the start index of the upper partition (inclusive)
 * @param range_starts Array to use for communication
 * @param range_sizes Array to use for communication
 * @param comm The communicator to use
 */
void partition_parallel(
        int *data, int datasize, int *p1, int *p2,
        int *range_starts, int *range_sizes,
        MPI_Comm comm)
{
    // Get rank and size
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Get range to work on
    int range_start = datasize * rank / size;
    int range_end = datasize * (rank+1) / size;
    int range_size = range_end - range_start;

    // Allocate buffer space
    int bufsize = size > range_size ? size : range_size;
    int* buf = (int*) malloc(bufsize * sizeof(int));

    // Get random ele from each process
    int pivot = data[range_start + rand()%range_size];

    // Get median of those eles to determine pivot
    MPI_Allgather(
            &pivot, 1, MPI_INTEGER,
            buf, 1, MPI_INTEGER,
            comm);
    pivot = quickselect(buf, size, size/2);

    // Partition the range
    int j = 0;
    int k = range_size;
    for (int i = 0; i < range_size; i++) {
        int num = data[range_start + i];
        if (num < pivot) buf[j++] = num;
        if (num > pivot) buf[--k] = num;
    }

    //// Combine lower partitions
    // Gather sizes of lower partitions
    MPI_Allgather(
            &j, 1, MPI_INTEGER,
            range_sizes, 1, MPI_INTEGER,
            comm);

    // Determine positions of lower partitions
    int a = 0;
    for (int i = 0; i < size; i++) {
        range_starts[i] = a;
        a += range_sizes[i];
    }

    // Gather lower partitions together
    MPI_Allgatherv(
            buf, j, MPI_INTEGER,
            data, range_sizes, range_starts, MPI_INTEGER,
            comm);

    //// Combine upper partitions
    // Gather sizes of upper partitions
    int usize = range_size - k;
    MPI_Allgather(
            &usize, 1, MPI_INTEGER,
            range_sizes, 1, MPI_INTEGER,
            comm);

    // Determine positions of upper partitions
    int b = datasize;
    for (int i = size-1; i >= 0; i--) {
        b -= range_sizes[i];
        range_starts[i] = b;
    }

    // Gather upper partitions together
    MPI_Allgatherv(
            buf+k, usize, MPI_INTEGER,
            data, range_sizes, range_starts, MPI_INTEGER,
            comm);

    // Fill remaining data between the partitions
    for (int i = a; i < b; i++)
        data[i] = pivot;

    // Assign outputs
    *p1 = a;
    *p2 = b;

    free(buf);
}

/**
 * @brief Sorts an array using quicksort in parallel. After this operation, the
 *        process with rank 0 in will contain the sorted array
 *
 * @param data The array to sort
 * @param datasize The size of the array
 * @param range_starts Array used for communication
 * @param range_sizes Array used for communication
 * @param comm The communicator to use
 */
void quicksort_parallel(
        int *data, int datasize,
        int *range_starts, int *range_sizes,
        MPI_Comm comm)
{
    // Get rank and size
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Perform remaining sort serially if group size 1
    if (size <= 1) {
        quicksort_serial(data, datasize);
        return;
    }

    // Partition the array
    int p1, p2;
    partition_parallel(
            data, datasize, &p1, &p2,
            range_starts, range_sizes, comm);

    // If lower partition is empty, only sort upper partition
    if (p1 == 0) {
        quicksort_parallel(
                data+p2, datasize-p2,
                range_starts, range_sizes,
                comm);
        return;
    }

    // If upper partition is empty, only sort lower partition
    if (p2 == datasize) {
        quicksort_parallel(
                data, p1,
                range_starts, range_sizes,
                comm);
        return;
    }

    // Compute remaining unsorted elements and exit early if no unsorted elements remain
    int num_unsorted = datasize - p2 + p1;
    if (num_unsorted == 0)
        return;

    // Determine how to split threads proportonally between the partitions
    int split = (size*p1 + num_unsorted/2) / num_unsorted;

    // Ensure each group has at least one thread
    if (split < 1)
        split = 1;
    if (split >= size)
        split = size-1;

    // Split the process group
    MPI_Comm newcomm;
    int color = rank < split;
    MPI_Comm_split(comm, color, rank, &newcomm);

    // Perform actions depending on process group
    if (color) {
        // First group sorts lower partition
        quicksort_parallel(
                data, p1,
                range_starts, range_sizes,
                newcomm);

        // First process of first group recieves data
        if (rank == 0) {
            MPI_Recv(
                    data+p2, datasize-p2, MPI_INTEGER,
                    split, 0, comm, MPI_STATUS_IGNORE);
        }
    } else {
        // Second group sorts upper partition
        quicksort_parallel(
                data+p2, datasize-p2,
                range_starts+split, range_sizes+split,
                newcomm);

        // First process of second group sends data
        if (rank == split) {
            MPI_Send(
                    data+p2, datasize-p2, MPI_INTEGER,
                    0, 0, comm);
        }
    }
}

int main(int argc, char** argv)
{
    int *data;
    int datasize;

    int next;
    int *range_starts;
    int *range_sizes;
    int range_start, range_end, range_size;

    int rank, size;

    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Exit if wrong number of arguments
    if (argc < 2 || argc > 4) {
        // Not shown here, but outfile may be omitted to save time writing output
        // Also, additional argument for unsorted output can be included for testing
        if (rank == 0)
            printf("Usage: mpirun -np <workers> %s <size> <outfile>\n", argv[0]);

        MPI_Finalize();
        return 0;
    }

    // Read size and exit if invalid
    datasize = atoi(argv[1]);
    if (datasize == 0) {
        if (rank == 0)
            printf("Usage: mpirun -np <workers> %s <size> <outfile>\n", argv[0]);

        MPI_Finalize();
        return 0;
    }

    // Initialize memory
    data = (int*) malloc(datasize * sizeof(int));
    range_starts = (int*) malloc(size * sizeof(int));
    range_sizes = (int*) malloc(size * sizeof(int));

    // Get ranges for each process
    next = 0;
    for (int i = 0; i < size; i++) {
        range_starts[i] = next;
        next = datasize * (i+1) / size;
        range_sizes[i] = next - range_starts[i];
    }

    // Get range for this process
    range_start = range_starts[rank];
    range_size = range_sizes[rank];
    range_end = range_start + range_size;

    // Seed random number generation. We use rank+1,
    // since seeds 0 and 1 produce the same sequences
    srand(rank+1); 

    // Generate data on range
    for (int i = range_start; i < range_end; i++)
        data[i] = rand() % datasize;

    // Gather each worker's data
    MPI_Allgatherv(
            MPI_IN_PLACE, range_size, MPI_INTEGER,
            data, range_sizes, range_starts, MPI_INTEGER,
            MPI_COMM_WORLD);

    // If master process, write unsorted data if necessary
    if (rank == 0 && argc == 4)
            print_numbers(argv[3], (uint32_t*) data, datasize);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = monotonic_seconds();
    // BEGIN TIMING ------------------------------------------------------------

    quicksort_parallel(
            data, datasize,
            range_starts, range_sizes,
            MPI_COMM_WORLD);

    // END TIMING --------------------------------------------------------------
    end_time = monotonic_seconds();

    // Write output if master process
    if (rank == 0) {
        print_time(end_time - start_time);

        // Write sorted list to output file if provided
        if (argc >= 3)
            print_numbers(argv[2], (uint32_t*) data, datasize);
    }

    free(data);
    free(range_starts);
    free(range_sizes);
    MPI_Finalize();
    return 0;
}

