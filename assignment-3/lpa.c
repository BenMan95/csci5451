// #include <stdio.h>
// #include <stdlib.h>
#include <mpi.h>

#include "utils.c"

int main(int argc, char** argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Clean up
    free_graph(&graph);
    free(range_sizes);
    free(range_starts);
    MPI_Finalize();
    return 0;
}
