#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "utils.h"

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
 * @brief Output the seconds elapsed while execution.
 *
 * @param seconds Seconds spent on execution, excluding IO.
 */
static void print_time(double const seconds)
{
  printf("Execution time: %0.04fs\n", seconds);
}

// Global variables used for thread control
int count;
int threads;
pthread_mutex_t lock;
pthread_barrier_t barrier;

// Global variables that are part of the algorithm
int iterations;
int num_points; // n
int dim_points; // m
double* points; // X
double* labels; // y
double* denom_vals; // Denominator values
double* XT; // X^T
double* Xw; // X*w
double* w;

double numerator;
double loss;

/**
 * @brief Clean up global variables
 *
 * @param step Designates which variables need to be freed
 *             0 - points and labels
 *             1 - points, labels, denom_vals, XT, Xw, and w
 *             2 - Previous vars and lock
 */
void cleanup(int step)
{
    free(points);
    free(labels);

    if (step < 1) {
        return;
    }

    free(denom_vals);
    free(XT);
    free(Xw);
    free(w);

    if (step < 2) {
        return;
    }

    pthread_mutex_destroy(&lock);

    if (step < 3) {
        return;
    }

    pthread_barrier_destroy(&barrier);
}

void* set_ranges(int* start1, int* end1, int* start2, int* end2)
{
    // Get index of thread
    pthread_mutex_lock(&lock);
    int i = count++;
    pthread_mutex_unlock(&lock);

    // Determine range to iterate over for dim_points
    int a = dim_points / threads;
    int b = dim_points % threads;
    *start1 = a*i + (i<b ? i : b);
    *end1 = *start1 + a + (i<b);

    // Determine range to iterate over for num_points
    int c = num_points / threads;
    int d = num_points % threads;
    *start2 = c*i + (i<d ? i : d);
    *end2 = *start2 + c + (i<d);
}

void* thread_func(void* is_main)
{
    // Get ranges to perform work on
    int start1;
    int end1;
    int start2;
    int end2;
    set_ranges(&start1, &end1, &start2, &end2);

    // Compute X^T and denominator values
    // Initialize w to all 0s
    for (int i = start1; i < end1; i++) {
        double sum = 0;
        double* ptr = points + i;
        double* row = XT + i*num_points;
        for (int j = 0; j < num_points; j++) {
            row[j] = *ptr;
            sum += (*ptr) * (*ptr);
            ptr += dim_points;
        }

        denom_vals[i] = sum;
        w[i] = 0;
    }

    // Initialize Xw to all 0s
    for (int i = start2; i < end2; i++) {
        Xw[i] = 0;
    }

    loss = 0;

    // Apply the algorithm
    for (int iter = 0; iter < iterations; iter++) {
        
        for (int i = 0; i < dim_points; i++) {

            // Compute new w_i value if denominator is not 0
            // Otherwise, leave w_i as 0
            if (denom_vals[i] == 0) {
                continue;
            }

            if (is_main) {
                numerator = 0;
            }

            double *row = XT + i*num_points;

            // Wait until initialization/previous step finished
            pthread_barrier_wait(&barrier);

            // Compute sum
            double sum = 0;
            for (int j = start2; j < end2; j++) {
                // The jth number in X_-i * w_-i
                double Xw_minusi = Xw[j] - (row[j])*w[i];

                sum += row[j] * (labels[j] - Xw_minusi);
            }

            // Add sum to numerator
            pthread_mutex_lock(&lock);
            numerator += sum;
            pthread_mutex_unlock(&lock);

            // Wait until numerator computed
            pthread_barrier_wait(&barrier);

            // Update X*w vector
            double wi_new = numerator / denom_vals[i];
            for (int j = start2; j < end2; j++) {
                Xw[j] += row[j] * (wi_new - w[i]);
            }

            // Wait until Xw updated
            pthread_barrier_wait(&barrier);

            if (is_main) {
                w[i] = numerator / denom_vals[i];
            }
        }

        // Compute sum
        double sum = 0;
        for (int i = start2; i < end2; i++) {
            double diff = Xw[i] - labels[i];
            sum += diff * diff;
        }

        // Add sum to loss
        pthread_mutex_lock(&lock);
        loss += sum;
        pthread_mutex_unlock(&lock);

        pthread_barrier_wait(&barrier);
        if (is_main) {
            printf("Iteration %d loss: %lf\n", iter+1, loss);
            loss = 0;
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("Usage: %s <points> <labels> <iters> <threads>\n", argv[0]);
        return -1;
    }

    iterations = atoi(argv[3]);
    if (iterations == 0) {
        perror("atoi");
        return -1;
    }

    threads = atoi(argv[4]);
    if (threads == 0) {
        perror("atoi");
        return -1;
    }

    int result;

    result = load_data(argv[1], argv[2],
                       &num_points, &dim_points,
                       &points, &labels);
    if (result) {
        return -1;
    }

    result = allocate_space(num_points, dim_points,
                            &denom_vals, &XT, &Xw, &w);
    if (result) {
        cleanup(0);
        return -1;
    }

    result = pthread_mutex_init(&lock, NULL);
    if (result) {
        perror("pthread_mutex_init");
        cleanup(1);
        return -1;
    }

    result = pthread_barrier_init(&barrier, NULL, threads);
    if (result) {
        perror("pthread_barrier_init");
        cleanup(2);
        return -1;
    }

    pthread_t* thread_handles = malloc((threads-1) * sizeof(thread_handles));
    if (thread_handles == NULL) {
        perror("malloc");
        cleanup(3);
        return -1;
    }

    // Begin timing
    double start_time = monotonic_seconds();

    // Create threads
    count = 0;
    for (int i = 0; i < threads-1; i++) {
        int is_main = 0;
        pthread_create(&thread_handles[i], NULL, thread_func, NULL);
    }

    // Perform work in main thread
    thread_func(NULL + 1);

    // Join threads
    for (int i = 0; i < threads-1; i++) {
        pthread_join(thread_handles[i], NULL);
    }

    // Finish timing
    double end_time = monotonic_seconds();
    printf("\n");
    print_time(end_time - start_time);

    printf("\nFinal weights:\n");
    for (int i = 0; i < dim_points; i++) {
        printf("%d: %lf\n", i, w[i]);
    }

    cleanup(3);
    free(thread_handles);

    return 0;
}

