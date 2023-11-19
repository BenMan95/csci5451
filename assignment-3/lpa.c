#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "utils.c"

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

    // Determine ranges that each thread will work on
    int next = 0;
    for (int i = 0; i < size; i++) {
        range_starts[i] = next;
        next = graph.num_nodes * (i+1) / size;
        range_sizes[i] = next - range_starts[i];
    }

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
            displs[i] = graph.offsets[range_starts[i]];

            for (int j = 0; j < range_sizes[i]; j++) {
                sendcounts[i] += graph.counts[range_starts[i] + j];
            }
        }

        // Scatter edges
        MPI_Scatterv(
                graph.neighbors, sendcounts, displs, MPI_INT,
                MPI_IN_PLACE, sendcounts[0], MPI_INT,
                0, MPI_COMM_WORLD);

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
    // neighbors[offsets[i-range_start]+j]
    // is the jth neighbor of the node i

    // STEPS 2-5 ---------------------------------------------------------------
    // Seed random number generation
    srand(rank+1);

    // Initialize labels
    unsigned int *labels = (unsigned int*) malloc(graph.num_nodes * sizeof(int));
    for (int i = range_start; i < range_end; i++) {
        labels[i] = i;
    }

    // Create array for counts
    int *counts = (int*) malloc(graph.num_nodes * sizeof(int));

    // Iterate until convergence
    int converge = 0;
    while (!converge) {
        // Gather labels
        MPI_Allgatherv(
                MPI_IN_PLACE, range_size, MPI_INT,
                labels, range_sizes, range_starts, MPI_INT,
                MPI_COMM_WORLD);

        // Recheck labels and check convergence
        converge = 1;
        for (int i = 0; i < range_size; i++) {
            // First, initialize counts to zero
            for (int j = 0; j < graph.num_nodes; j++) {
                counts[j] = 0;
            }

            // Initialize count for current label to one
            counts[labels[range_start+i]] = 1;

            // Count instances of each label among neighbors
            int best = 1;
            for (int j = 0; j < graph.counts[i]; j++) {
                int neighbor = graph.neighbors[graph.offsets[i]+j];
                int label = labels[neighbor];
                counts[label]++;

                // Track highest count of same label
                if (counts[label] > best) {
                    best = counts[label];
                }
            }

            // Count labels that have the highest count
            int amtbest = 0;
            for (int j = 0; j < graph.num_nodes; j++) {
                if (counts[j] == best) {
                    amtbest++;
                }
            }

            // Randomly pick a label with the highest count
            int n = rand() % amtbest;

            // Find the nth label with that count
            for (int j = 0; j < graph.num_nodes; j++) {
                if (counts[j] == best) {
                    if (n == 0) {
                        // Set not converged if the label could have changed
                        if (amtbest > 1 || labels[i+range_start] != j) {
                            converge = 0;
                        }

                        // Set the new label
                        labels[i+range_start] = labels[j];
                        break;
                    } else {
                        n--;
                    }
                }
            }
        }

        // Check if values have converged in all processes
        MPI_Allreduce(
            MPI_IN_PLACE, &converge, 1,
            MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    // Output labels
    if (rank == 0) {
        print_labels(argv[2], labels, graph.num_nodes);
    }

    // Clean up
    free_graph(&graph);
    free(range_sizes);
    free(range_starts);
    MPI_Finalize();
    return 0;
}
