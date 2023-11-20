#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "utils.c"

/**
 * @brief (Unused) Splits the nodes of a graph into n ranges such
 *        that each range has approximately the same unused of nodes
 *
 * @param graph The graph to split
 * @param range_starts Where to write the starting indices of the ranges
 * @param range_sizes Where to write the sizes of the ranges
 * @param n The number of ranges to create
 */
void split_by_nodes(graph_t *graph, int *range_starts, int *range_sizes, int n)
{
    int next = 0;
    for (int i = 0; i < size; i++) {
        range_starts[i] = next;
        next = graph.num_nodes * (i+1) / size;
        range_sizes[i] = next - range_starts[i];
    }
}

/**
 * @brief Splits the nodes of a graph into n ranges by their neighbor counts
 *        Rather than splitting so that each range has the same amount of nodes,
 *        This splits the graph so that the total number of neighbors of its nodes
 *        is approximately the same
 *
 * @param graph The graph to split
 * @param range_starts Where to write the starting indices of the ranges
 * @param range_sizes Where to write the sizes of the ranges
 * @param n The number of ranges to create
 */
void split_by_neighbors(graph_t *graph, int *range_starts, int *range_sizes, int n)
{
    // First, the target neighbor count per range is computed
    // as twice the number of edges divided by the number of ranges
    // Then, iterate over the subarray while adding the neighber counts up
    // When the sum is close enough to the target, create a new subarray

    double target = 2 * graph->num_edges / n;
    double cur_sum = 0;
    double cur_diff = target;

    int prev_start = 0;
    int cur_range = 0;
    for (int i = 0; i < graph->num_nodes; i++) {
        // Compute new sum and the difference between it and the target
        double next_sum = cur_sum + graph->counts[i];
        double next_diff = next_sum - target;
        if (next_diff < 0)
            next_diff = -next_diff;

        // If adding another element would move away from the target,
        // create the next subarray
        if (next_diff > cur_diff) {
            // Set the values for this subarray
            range_starts[cur_range] = prev_start;
            range_sizes[cur_range] = i - prev_start;
            prev_start = i;
            cur_range++;

            // Recompute sum and diff
            next_sum -= target;
            next_diff -= target;
            if (next_diff < 0)
                next_diff = -next_diff;

            // Break early at last range
            if (cur_range+1 == n)
                break;
        }

        // Set sum and diff for next iteration
        cur_sum = next_sum;
        cur_diff = next_diff;
    }

    // Fill in remaining ranges. Usually only one range is left.
    // However, there can be more ranges if n is greater than the node count.
    for (; cur_range < n; cur_range++) {
        range_starts[cur_range] = prev_start;
        range_sizes[cur_range] = graph->num_nodes - prev_start;
        prev_start = graph->num_nodes;
    }
}

int main(int argc, char** argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Exit if wrong number of arguments
    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: mpirun -np <workers> %s <graph> <labels>\n", argv[0]);
        }

        MPI_Finalize();
        return 0;
    }

    // STEP 1 --------------------------------------------------------------------
    // First process loads graph from file
    graph_t graph;
    if (rank == 0) {
        load_graph(argv[1], &graph);
    }

    // Graph size is distributed among other processes
    MPI_Bcast(&graph.num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&graph.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate space for ranges
    int *range_starts = (int*) malloc(size * sizeof(int));
    int *range_sizes = (int*) malloc(size * sizeof(int));

    // First process determines what range each process will work on
    if (rank == 0) {
        split_by_neighbors(&graph, range_starts, range_sizes, size);
        // split_by_nodes(&graph, range_starts, range_sizes, size);
    }

    // Ranges are broadcasted to other processes
    MPI_Bcast(range_starts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(range_sizes, size, MPI_INT, 0, MPI_COMM_WORLD);

    // Get this thread's range
    int range_start = range_starts[rank];
    int range_size = range_sizes[rank];
    int range_end = range_start + range_size;

    if (rank == 0) { // First process scatters graph data
        // Scatter neighbor counts
        MPI_Scatterv(
                graph.counts, range_sizes, range_starts, MPI_INT,
                MPI_IN_PLACE, range_size, MPI_INT,
                0, MPI_COMM_WORLD);

        // Determine what edges to send to each process
        int *sendcounts = (int*) malloc(size * sizeof(int));
        int *displs = (int*) malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            sendcounts[i] = 0;
            for (int j = 0; j < range_sizes[i]; j++) {
                sendcounts[i] += graph.counts[range_starts[i] + j];
            }

            displs[i] = graph.offsets[range_starts[i]];
        }

        // Scatter edges
        MPI_Scatterv(
                graph.neighbors, sendcounts, displs, MPI_INT,
                MPI_IN_PLACE, sendcounts[0], MPI_INT,
                0, MPI_COMM_WORLD);

        // Clean up
        free(sendcounts);
        free(displs);
    } else { // Remaining processes recieve graph data
        // Recieve neighbor counts
        graph.counts = (int*) malloc(range_size * sizeof(int));
        MPI_Scatterv(
                NULL, NULL, NULL, MPI_INT,
                graph.counts, range_size, MPI_INT,
                0, MPI_COMM_WORLD);

        // Compute offsets and number of edges to recieve
        graph.offsets = (int*) malloc(range_size * sizeof(int));
        int count = 0;
        for (int i = 0; i < range_size; i++) {
            graph.offsets[i] = count;
            count += graph.counts[i];
        }

        // Recieve edges
        graph.neighbors = (int*) malloc(count * sizeof(int));
        MPI_Scatterv(
                NULL, NULL, NULL, MPI_INT,
                graph.neighbors, count, MPI_INT,
                0, MPI_COMM_WORLD);
    }

    // After graph data is scattered,
    // counts[i-range_start] is the number of neighbors of node i
    // neighbors[offsets[i-range_start]+j] is the jth neighbor of the node i

    // STEPS 2-5 ---------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Initialize labels
    unsigned int *labels = (unsigned int*) malloc(graph.num_nodes * sizeof(int));
    for (int i = range_start; i < range_end; i++) {
        labels[i] = i;
    }

    // STEP 5 ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Iterate until convergence
    int converge = 0;
    while (!converge) {
        // Gather labels from each process
        MPI_Allgatherv(
                MPI_IN_PLACE, range_size, MPI_INT,
                labels, range_sizes, range_starts, MPI_INT,
                MPI_COMM_WORLD);

        // Recheck labels and check convergence
        converge = 1;
        for (int i = 0; i < range_size; i++) {
            int cur_label = labels[range_start+i];

            // Find the highest label among self and neighbors
            int max_label = cur_label;
            for (int j = 0; j < graph.counts[i]; j++) {
                int neighbor = graph.neighbors[graph.offsets[i]+j];
                int label = labels[neighbor];
                if (label > max_label) {
                    max_label = label;
                }
            }

            // Set new label and check convergence
            if (max_label != cur_label) {
                labels[range_start+i] = max_label;
                converge = 0;
            }
        }

        // Check convergence on all processes
        MPI_Allreduce(
            MPI_IN_PLACE, &converge, 1,
            MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    // ALGORITHM FINISHED ------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Print time and labels
    if (rank == 0) {
        print_time25(t2-t0);
        print_time5(t2-t1);
        print_labels(argv[2], labels, graph.num_nodes);
    }

    // Clean up
    free_graph(&graph);
    free(range_sizes);
    free(range_starts);
    MPI_Finalize();
    return 0;
}
