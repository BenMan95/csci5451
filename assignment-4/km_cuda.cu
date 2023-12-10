#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define MAX_ITERS 20

/* Gives us high-resolution timers. */
#undef _POSIX_C_SOURCE // This prevents the compiler displaying a warning
#define _POSIX_C_SOURCE 199309L
#include <time.h>

/* OSX timer includes */
#ifdef __MACH__
    #include <mach/mach.h>
    #include <mach/mach_time.h>
#endif

/**
 * @brief Return the number of seconds since an unspecified time (e.g., Unix
 *        epoch). This is accomplished with a high-resolution monotonic timer,
 *        suitable for performance timing.
 *
 * @return The number of seconds.
 */
static inline double monotonic_seconds()
{
#ifdef __MACH__
    /* OSX */
    static mach_timebase_info_data_t info;
    static double seconds_per_unit;
    if(seconds_per_unit == 0) {
      mach_timebase_info(&info);
      seconds_per_unit = (info.numer / info.denom) / 1e9;
    }
    return seconds_per_unit * mach_absolute_time();
#else
    /* Linux systems */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/**
 * @brief Output the seconds elapsed while clustering.
 *
 * @param seconds Seconds spent on k-means clustering, excluding IO.
 */
static void print_time(double const seconds)
{
    printf("k-means clustering time: %0.04fs\n", seconds);
}

typedef struct {
    int num; // The number of points
    int dim; // The number of dimensions
    double* coords; // The jth coord of the ith points is coords[i*dim+j]
} points_t;

/**
 * @brief Loads point data from a file
 * 
 * @param filename The file to load point data from
 * @param points The points_t struct to load data to
 */
void load_points(
    char* filename,
    points_t* points)
{
    FILE *fh = fopen(filename, "r");
    fscanf(fh, "%d %d", &points->num, &points->dim);

    int size = points->num * points->dim;
    points->coords = (double*) malloc(size * sizeof(double));

    for (int i = 0; i < size; i++) {
        fscanf(fh, "%lf", points->coords + i);
    }

    fclose(fh);
}

/**
 * @brief Writes cluster assignments to clusters.txt
 * 
 * @param clusters The array to read cluster assignments from
 * @param n The number of points
 */
void output_clusters(int* clusters, int n)
{
    FILE *fh = fopen("clusters.txt", "w");

    for (int i = 0; i < n; i++) {
        fprintf(fh, "%d\n", clusters[i]);
    }

    fclose(fh);
}

/**
 * @brief Output centroids to centroids.txt
 * 
 * @param centroids The points_t struct to read centroids from
 */
void output_centroids(points_t centroids)
{
    FILE *fh = fopen("centroids.txt", "w");

    fprintf(fh, "%d %d\n", centroids.num, centroids.dim);
    for (int i = 0; i < centroids.num; i++) {
        for (int j = 0; j < centroids.dim; j++) {
            if (j != 0) {
                fprintf(fh, " ");
            }
            fprintf(fh, "%lf", centroids.coords[i*centroids.dim + j]);
        }
        fprintf(fh, "\n");
    }

    fclose(fh);
}

/**
 * @brief Determine the nearest centroid to a point
 * 
 * @param centroids The list of centroids to use
 * @param point The point to determine the nearest centroid for
 *              The dimension should match the dimension of the centroids list
 * 
 * @returns The index of the nearest centroid
 */
int nearest_centroid(points_t centroids, double* point)
{
    double best_dist = DBL_MAX; // Uses squared distance
    int best_centroid;

    for (int i = 0; i < centroids.num; i++) {
        // Compute distance to the ith centroid
        double dist = 0;
        for (int j = 0; j < centroids.dim; j++) {
            double diff = point[j] - centroids.coords[i*centroids.dim + j];
            dist += diff * diff;
        }

        // Update the best distance and centroid
        if (dist < best_dist) {
            best_dist = dist;
            best_centroid = i;
        }
    }

    return best_centroid;
}

/**
 * @brief Computes new centroids based on current clusters
 * 
 * @param centroids Where to write the new centroids to
 * @param points Where to read the points from
 * @param clusters The current cluster each point is assigned to
 * @param counts An array to be used for temporary storage
 */
void compute_centroids(points_t centroids, points_t points, int* clusters, int* counts)
{
    // Initialize counts/centroids to 0
    for (int i = 0; i < centroids.num; i++) {
        counts[i] = 0;
        for (int j = 0; j < centroids.dim; j++) {
            int idx = i*centroids.dim + j;
            centroids.coords[idx] = 0;
        }
    }

    // Compute coordinate sums and counts for each cluster
    for (int i = 0; i < points.num; i++) {
        int cluster = clusters[i];
        counts[cluster]++;
        for (int j = 0; j < points.dim; j++) {
            int c_idx = cluster*centroids.dim + j;
            int p_idx = i*points.dim + j;
            centroids.coords[c_idx] += points.coords[p_idx];
        }
    }

    // Divide to get averages
    for (int i = 0; i < centroids.num; i++) {
        int count = counts[i];
        for (int j = 0; j < centroids.dim; j++) {
            int idx = i*centroids.dim + j;
            centroids.coords[idx] = centroids.coords[idx] / count;
        }
    }
}

/**
 * @brief Assigns clusters for a set of points and centroids
 * 
 * @param points The points to assign clusters for
 * @param centroids The centroids to assigne clusters to
 * @param clusters Where to write cluster assignments to
 * 
 * @returns If cluster assignments have converged
 */
int assign_clusters(points_t points, points_t centroids, int* clusters)
{
    int converged = 1;
    for (int i = 0; i < points.num; i++) {
        // Determine new cluster
        int cluster = nearest_centroid(centroids, points.coords + i*points.dim);

        // Update cluster if necessary
        if (cluster != clusters[i]) {
            converged = 0;
            clusters[i] = cluster;
        }
    }
    return converged;
}


/**
 * @brief Determine the nearest centroid to a point
 * 
 * @param centroids The list of centroids to use
 * @param point The point to determine the nearest centroid for
 *              The dimension should match the dimension of the centroids list
 * 
 * @returns The index of the nearest centroid
 */
__device__ int d_nearest_centroid(points_t centroids, double* point)
{
    double best_dist = DBL_MAX; // Uses squared distance
    int best_centroid;

    for (int i = 0; i < centroids.num; i++) {
        // Compute distance to the ith centroid
        double dist = 0;
        for (int j = 0; j < centroids.dim; j++) {
            double diff = point[j] - centroids.coords[i*centroids.dim + j];
            dist += diff * diff;
        }

        // Update the best distance and centroid
        if (dist < best_dist) {
            best_dist = dist;
            best_centroid = i;
        }
    }

    return best_centroid;
}

// Initialize centroids
__global__ void init_centroids(points_t points,
                               points_t centroids)
{
    int stride = gridDim.x * blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int c_size = centroids.num * centroids.dim;
    int p_size = points.num * points.dim;
    for (int i = index; i < c_size; i += stride) {
        centroids.coords[i] = points.coords[i % p_size];
    }
}

// Initialize clusters
__global__ void init_clusters(points_t points,
                              points_t centroids,
                              int* clusters)
{
    int stride = gridDim.x * blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = index; i < points.num; i += stride) {
        double* point_idx = points.coords + i*points.dim;
        clusters[i] = d_nearest_centroid(centroids, point_idx);
    }
}
// Recompute centroids
__global__ void d_compute_centroids(points_t points,
                                    points_t centroids,
                                    int* clusters,
                                    int* counts)
{
    // Determine centroid range this block will cover
    int tot_threads = gridDim.x * blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int range_start = centroids.num * index / tot_threads;
    int range_end = centroids.num * (index+1) / tot_threads;

    // Initialize counts/centroids to 0
    for (int i = range_start; i < range_end; i++) {
        counts[i] = 0;
        for (int j = 0; j < centroids.dim; j++) {
            centroids.coords[i*centroids.dim + j] = 0;
        }
    }

    // Compute coordinate sums and counts for each cluster in range
    for (int i = 0; i < points.num; i++) {
        int cluster = clusters[i];

        // Update coordinates and counts if in the cluster range
        if (cluster >= range_start && cluster < range_end) {
            counts[cluster]++;
            for (int j = 0; j < points.dim; j++) {
                int c_idx = cluster*centroids.dim + j;
                int p_idx = i*points.dim + j;
                centroids.coords[c_idx] += points.coords[p_idx];
            }
        }
    }

    // Divide to get averages
    for (int i = range_start; i < range_end; i++) {
        int count = counts[i];
        for (int j = 0; j < centroids.dim; j++) {
            int idx = i*centroids.dim + j;
            centroids.coords[idx] = centroids.coords[idx] / count;
        }
    }
}

__global__ void d_assign_clusters(points_t points,
                                  points_t centroids,
                                  int* clusters,
                                  char* converged)
{
    int stride = gridDim.x * blockDim.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = index; i < points.num; i += stride) {
        // Determine new cluster
        int cluster = d_nearest_centroid(centroids, points.coords + i*points.dim);

        // Update cluster if necessary
        if (cluster != clusters[i]) {
            *converged = 0;
            clusters[i] = cluster;
        }
    }
}

int main(int argc, char** argv)
{
    // Check arguments
    if (argc != 5) {
        printf("Usage: %s <input> <clusters> <blocks> <threads>\n", argv[0]);
        return -1;
    }

    int blocks = atoi(argv[3]);
    int threads = atoi(argv[4]);

    // LOAD / INITIALIZE DATA --------------------------------------------------

    // Load point data
    points_t points;
    load_points(argv[1], &points);
    int p_size = points.num * points.dim;

    // Initialize centroids struct
    points_t centroids;
    centroids.num = atoi(argv[2]);
    centroids.dim = points.dim;
    int c_size = centroids.num * centroids.dim;
    centroids.coords = (double*) malloc(c_size * sizeof(double));

    // Allocate arrays for cluster assignments and counts
    int* clusters = (int*) malloc(points.num * sizeof(int));
    int* counts = (int*) malloc(centroids.num * sizeof(int));

    // ASSIGN DEVICE MEMORY ----------------------------------------------------

    // Copy points to device memory
    points_t d_points;
    d_points.num = points.num;
    d_points.dim = points.dim;
    cudaMalloc((void**) &d_points.coords, p_size * sizeof(double));
    cudaMemcpy(d_points.coords, points.coords, p_size * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate struct/device memory for centroids
    points_t d_centroids;
    d_centroids.num = centroids.num;
    d_centroids.dim = centroids.dim;
    cudaMalloc((void**) &d_centroids.coords, c_size * sizeof(double));

    int *d_clusters, *d_counts;
    cudaMalloc((void**) &d_clusters, points.num * sizeof(int));
    cudaMalloc((void**) &d_counts, centroids.num * sizeof(int));

    char *d_converged;
    cudaMalloc((void**) &d_converged, sizeof(char));

    // PERFORM ALGORITHM -------------------------------------------------------
    double t0 = monotonic_seconds();

    // Initialize centroids and clusters
    init_centroids<<<blocks, threads>>>(d_points, d_centroids);
    init_clusters<<<blocks, threads>>>(d_points, d_centroids, d_clusters);

    // Perform algorithm until convergence or iteration limit
    for (int i = 0; i < MAX_ITERS; i++) {
        d_compute_centroids<<<blocks, threads>>>(d_points, d_centroids, d_clusters, d_counts);

        // Assume convergence
        char converged = 1;
        cudaMemcpy(d_converged, &converged, sizeof(char), cudaMemcpyHostToDevice);

        // Reassign clusters
        d_assign_clusters<<<blocks, threads>>>(d_points, d_centroids, d_clusters, d_converged);
        cudaDeviceSynchronize();

        // Break early if converged
        cudaMemcpy(&converged, d_converged, sizeof(char), cudaMemcpyDeviceToHost);
        if (converged) {
            break;
        }
    }

    // OUTPUT RESULTS / CLEAN UP -----------------------------------------------
    double t1 = monotonic_seconds();
    print_time(t1 - t0);

    // Copy results from device memory to host memory
    cudaMemcpy(centroids.coords, d_centroids.coords, c_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(clusters, d_clusters, points.num * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    output_clusters(clusters, points.num);
    output_centroids(centroids);

    // Free memory
    free(points.coords);
    free(centroids.coords);
    free(clusters);
    free(counts);
    cudaFree(d_points.coords);
    cudaFree(d_centroids.coords);
    cudaFree(d_clusters);
    cudaFree(d_counts);
    cudaFree(d_converged);

    return 0;
}
