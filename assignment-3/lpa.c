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
    for (int i = 0; i < n; i++) {
        range_starts[i] = next;
        next = graph->num_nodes * (i+1) / n;
        range_sizes[i] = next - range_starts[i];
    }
}

/**
 * @brief Splits the nodes of a graph into n ranges by their edge counts
 *        Rather than splitting so that each range has the same amount of nodes,
 *        This splits the graph so that the total number of edges of its nodes
 *        is approximately the same
 *
 * @param graph The graph to split
 * @param range_starts Where to write the starting indices of the ranges
 * @param range_sizes Where to write the sizes of the ranges
 * @param n The number of ranges to create
 */
void split_by_edges(graph_t *graph, int *range_starts, int *range_sizes, int n)
{
    // First, the target edge count per range is computed as
    // the number of edges (both directions) divided by the number of ranges
    // Then, iterate over the subarray while adding the neighber counts up
    // When the sum is close enough to the target, create a new subarray

    double target = graph->num_edges / n;
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

/**
 * @brief Searches a sorted array using binary search
 * 
 * @param arr The array to search
 * @param n The size of the array
 * @param x The element to search for
 * 
 * @return Returns the index if found, -1 otherwise
 */
int binary_search(int *arr, int n, int x)
{
    if (n <= 0) return -1;

    int k = n / 2;
    int mid = arr[k];

    if (x < mid) return binary_search(arr, k, x);
    if (x > mid) return binary_search(arr+k+1, n-k-1, x) + k + 1;
    return k;
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
    graph_t full_graph;
    if (rank == 0) {
        load_graph(argv[1], &full_graph);
    }

    // Graph size is broadcasted to other processes
    MPI_Bcast(&full_graph.num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&full_graph.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate space for ranges
    int *range_starts = (int*) malloc(size * sizeof(int));
    int *range_sizes = (int*) malloc(size * sizeof(int));

    // First process determines what range each process will work on
    if (rank == 0) {
        // split_by_nodes(&full_graph, range_starts, range_sizes, size);
        split_by_edges(&full_graph, range_starts, range_sizes, size);
    }

    // Ranges are broadcasted to other processes
    MPI_Bcast(range_starts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(range_sizes, size, MPI_INT, 0, MPI_COMM_WORLD);

    // Get this thread's range
    int range_start = range_starts[rank];
    int range_size = range_sizes[rank];
    int range_end = range_start + range_size;

    graph_t local_graph = {.num_nodes = range_size, .num_edges = 0};

    // Scatter edge counts
    local_graph.counts = (int*) malloc(range_size * sizeof(int));
    MPI_Scatterv(
        full_graph.counts, range_sizes, range_starts, MPI_INT,
        local_graph.counts, range_size, MPI_INT,
        0, MPI_COMM_WORLD);

    // Each process computes offsets and number of edges to recieve
    local_graph.offsets = (int*) malloc(range_size * sizeof(int));
    for (int i = 0; i < range_size; i++) {
        local_graph.offsets[i] = local_graph.num_edges;
        local_graph.num_edges += local_graph.counts[i];
    }

    // First process also determines what edges to send to each process
    int *sendcounts;
    int *displs;
    if (rank == 0) {
        sendcounts = (int*) malloc(size * sizeof(int));
        displs = (int*) malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            sendcounts[i] = 0;
            for (int j = 0; j < range_sizes[i]; j++) {
                sendcounts[i] += full_graph.counts[range_starts[i]+j];
            }
            displs[i] = full_graph.offsets[range_starts[i]];
        }
    }

    // Scatter edges
    local_graph.edges = (int*) malloc(local_graph.num_edges * sizeof(int));
    MPI_Scatterv(
        full_graph.edges, sendcounts, displs, MPI_INT,
        local_graph.edges, local_graph.num_edges, MPI_INT,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(sendcounts);
        free(displs);

        // Edge data of full graph is freed after use
        // However, the num_nodes and num_edges values remain for later use
        free_graph(&full_graph);
    }

    // For local graph data,
    // local_graph.counts[i-range_start] is the number of edges of node i
    // local_graph.edges[local_graph.offsets[i-range_start]+j] is the jth edge of the node i

    // STEPS 2-5 ---------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // ranks[i] will be the rank of the process handling node i
    int *ranks = (int*) malloc(full_graph.num_nodes * sizeof(int));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < range_sizes[i]; j++) {
            ranks[range_starts[i]+j] = i;
        }
    }

    struct {
        int num_labels = 0;
        int *nodes; // The node that each label is for
        int *labels; // Array for labels to send
        int *counts;
        int *displs;
    } send_data;

    struct {
        int num_labels = 0; // The number of labels to recieve
        int *nodes; // The node that each label is for
        int *labels; // Array for labels to recieve
        int *counts;
        int *displs;
    } recv_data;

    // Assign arrays for counts and displacements
    send_data.counts = (int*) malloc(size * sizeof(int));
    send_data.displs = (int*) malloc(size * sizeof(int));
    recv_data.counts = (int*) malloc(size * sizeof(int));
    recv_data.displs = (int*) malloc(size * sizeof(int));

    // Initialize those arrays
    for (int i = 0; i < size; i++) {
        send_data.counts[i] = 0;
        send_data.displs[i] = 0;
        recv_data.counts[i] = 0;
        recv_data.displs[i] = 0;
    }

    // Initialize labels array
    // Not used for labels yet, just used now to avoid needing to allocate extra space
    int *temp1 = (int*) malloc(full_graph.num_nodes * sizeof(int));
    for (int i = 0; i < full_graph.num_nodes; i++) {
        temp1[i] = 0;
    }

    // Initialize indices to 0
    int *indices = (int*) malloc(size * sizeof(int)); // Current position in each subarray
    for (int i = 0; i < size; i++) {
        indices[i] = 0;
    }

    // First pass over edges array
    int *temp2 = (int*) malloc(size * sizeof(int)); // Used to check which processes are already being sent this node
    for (int i = 0; i < range_size; i++) {
        for (int j = 0; j < size; j++) {
            temp2[j] = 0;
        }

        for (int j = 0; j < local_graph.counts[i]; j++) {
            int edge = local_graph.edges[local_graph.offsets[i]+j];
            int edge_rank = ranks[edge];

            // Skip edges handled by the same process
            if (rank == edge_rank) {
                continue;
            }

            // Count the edges that will need to be sent to other processes
            if (temp2[edge_rank] == 0) {
                temp2[edge_rank] = 1;
                send_data.num_labels++;
                send_data.counts[edge_rank]++;
            }

            // Determine what edges will need to be recieved
            if (temp1[edge] == 0) {
                temp1[edge] = 1;
                recv_data.num_labels++;
            }
        }
    }

    // Assign memory space
    send_data.nodes = (int*) malloc(send_data.num_labels * sizeof(int));
    send_data.labels = (int*) malloc(send_data.num_labels * sizeof(int));
    recv_data.nodes = (int*) malloc(recv_data.num_labels * sizeof(int));
    recv_data.labels = (int*) malloc(recv_data.num_labels * sizeof(int));

    int idx = 0;
    for (int i = 0; i < size; i++) {
        // Compute displacements
        send_data.displs[i] = idx;
        idx += send_data.counts[i];
    }

    // Determine what labels will need to be sent to each process
    for (int i = 0; i < range_size; i++) {
        for (int j = 0; j < size; j++) {
            temp2[j] = 0;
        }

        for (int j = 0; j < local_graph.counts[i]; j++) {
            int edge = local_graph.edges[local_graph.offsets[i]+j];
            int edge_rank = ranks[edge];

            // Skip edges handled by the same process
            if (rank == edge_rank) {
                continue;
            }

            // Add edge to send data
            if (temp2[edge_rank] == 0) {
                temp2[edge_rank] = 1;
                int k = indices[edge_rank]++;
                send_data.nodes[send_data.displs[edge_rank]+k] = range_start+i;
            }
        }
    }

    // Determine what labels will need to be recieved from each process
    idx = 0;
    for (int i = 0; i < size; i++) {
        recv_data.displs[i] = idx;
        for (int j = 0; j < range_sizes[i]; j++) {
            int node = range_starts[i] + j;
            if (temp1[node]) {
                recv_data.nodes[idx++] = node;
                recv_data.counts[i]++;
            }
        }
    }

    // Initialize labels
    int *local_labels = (int*) malloc(range_size * sizeof(int));
    for (int i = 0; i < range_size; i++) {
        local_labels[i] = range_start + i;
    }

    // Free temporary arrays
    free(ranks);
    free(temp1);
    free(temp2);

    // STEP 5 ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Iterate until convergence
    int converge = 0;
    while (!converge) {
        // Load labels to send buffer
        for (int i = 0; i < send_data.num_labels; i++) {
            send_data.labels[i] = local_labels[send_data.nodes[i] - range_start];
        }

        // Exchange labels between processes
        MPI_Alltoallv(
            send_data.labels, send_data.counts, send_data.displs, MPI_INT,
            recv_data.labels, recv_data.counts, recv_data.displs, MPI_INT,
            MPI_COMM_WORLD);

        // Recheck labels and check convergence
        converge = 1;
        for (int i = 0; i < range_size; i++) {
            int cur_label = local_labels[i];

            // Find the highest label among self and edges
            int min_label = cur_label;
            for (int j = 0; j < local_graph.counts[i]; j++) {
                int edge = local_graph.edges[local_graph.offsets[i]+j];

                // Get label
                int label;
                if (edge < range_start || edge >= range_start+range_size) {
                    // Read from recieved labels if not in range
                    int k = binary_search(recv_data.nodes, recv_data.num_labels, edge);
                    label = recv_data.labels[k];
                } else {
                    // Else read from local labels
                    label = local_labels[edge - range_start];
                }

                if (label < min_label) {
                    min_label = label;
                }
            }

            // Set new label and check convergence
            if (min_label != cur_label) {
                local_labels[i] = min_label;
                converge = 0;
            }
        }

        // Check convergence on all processes
        MPI_Allreduce(
            MPI_IN_PLACE, &converge, 1,
            MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    }

    // Gather all labels at first process
    int* labels = NULL;
    if (rank == 0) {
        labels = (int*) malloc(full_graph.num_nodes * sizeof(int));
    }

    MPI_Gatherv(
        local_labels, range_size, MPI_INT,
        labels, range_sizes, range_starts, MPI_INT,
        0, MPI_COMM_WORLD);

    // ALGORITHM FINISHED ------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    // Print time and labels
    if (rank == 0) {
        print_time25(t2-t0);
        print_time5(t2-t1);
        print_labels(argv[2], (unsigned int*) labels, full_graph.num_nodes);
        free(labels);
    }

    // Free local graph data arrays
    free_graph(&local_graph);

    // Free send data arrays
    free(send_data.nodes);
    free(send_data.labels);
    free(send_data.counts);
    free(send_data.displs);

    // Free recieve data arrays
    free(recv_data.nodes);
    free(recv_data.labels);
    free(recv_data.counts);
    free(recv_data.displs);

    // Free ranges
    free(range_sizes);
    free(range_starts);

    // Free other arrays
    free(local_labels);

    MPI_Finalize();
    return 0;
}
