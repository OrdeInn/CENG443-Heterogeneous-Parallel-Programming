#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

// Number of vertices in the graph
#define MAX 9
#define INF 99999
#define THREADS_BLOCK 1


__global__ void findClosestVertice(int* distance, int* visited, int* global_closest, int num_vertices) {
    
    int dist = INF + 1;
    int vertice = -1;
    int i;

    for (i = 0; i < num_vertices; i++) {
        
        if ((distance[i] < dist) && (visited[i] != 1)) {
            
            dist = distance[i];
            vertice = i;
        }
    }

    global_closest[0] = vertice;
    visited[vertice] = 1;
}

__global__ void relaxEdges(int* graph, int* distance, int* parent_node, int* visited, int* global_closest) {

    int next = blockIdx.x * blockDim.x + threadIdx.x;
    int source = global_closest[0];

    int edge = graph[source * MAX + next];
    int new_dist = distance[source] + edge;

    if ((edge != 0) && (visited[next] != 1) && (new_dist < distance[next])) {

        distance[next] = new_dist;
        parent_node[next] = source;
    }

}



int main(){

	int n = 9;
	int start = 0;

	int* graph = (int*) malloc(MAX * MAX * sizeof(int));
	int* distance = (int*) malloc(MAX * sizeof(int));
	int* visited = (int*) malloc(MAX * sizeof(int));
    int* parent_vertice = (int*) malloc(MAX * sizeof(int));
	
    int Graph[9][9] = { { 0, 4, 0, 0, 0, 0, 0, 8, 0 },
                        { 4, 0, 8, 0, 0, 0, 0, 11, 0 },
                        { 0, 8, 0, 7, 0, 4, 0, 0, 2 },
                        { 0, 0, 7, 0, 9, 14, 0, 0, 0 },
                        { 0, 0, 0, 9, 0, 10, 0, 0, 0 },
                        { 0, 0, 4, 14, 10, 0, 2, 0, 0 },
                        { 0, 0, 0, 0, 0, 2, 0, 1, 6 },
                        { 8, 11, 0, 0, 0, 0, 1, 0, 7 },
                        { 0, 0, 2, 0, 0, 0, 6, 7, 0 } }; 


	for(int j=0 ; j < n; j++){

		for(int i = 0; i < n; i++){

			graph[i + (j*n)] = Graph[j][i];
		}
	}

	for (int i = 0; i < n; i++) {

        if(graph[i + (start*n)] == 0){
    
    		distance[i] = INF; 
        }
        else{

    		distance[i] = graph[i + (start*n)]; 
        }

		visited[i] = 0;
	}

    for(int i = 0; i < n; i++){

        parent_vertice[i] = -1;
    }

    distance[start] = 0;

	cudaError_t err = cudaSuccess;

	int size = MAX  * sizeof(int);

	int *d_graph = NULL;
    err = cudaMalloc((void **)&d_graph, MAX * MAX * sizeof(int));

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }

	int *d_distance = NULL;
    err = cudaMalloc((void **)&d_distance, size);

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }

	int *d_visited = NULL;
    err = cudaMalloc((void **)&d_visited, size);

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }

    int *d_closest = NULL;
    err = cudaMalloc((void **)&d_closest, sizeof(int));

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }

    int *d_parent_vertice = NULL;
    err = cudaMalloc((void **)&d_parent_vertice, size);

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(d_graph, graph, MAX * MAX * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_distance, distance, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_visited, visited, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_parent_vertice, parent_vertice, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	dim3 threads(1, 1);
    dim3 blocks(1, 1);

    dim3 gridRelax(MAX / THREADS_BLOCK, 1);
    dim3 blockRelax(THREADS_BLOCK, 1);


    //Measure execution Time
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    cudaEventRecord(e_start);


    for(int i = 0; i < n; i++){

        findClosestVertice<<<blocks, threads>>>(d_distance, d_visited, d_closest, n);
        relaxEdges<<<gridRelax, blockRelax>>>(d_graph, d_distance, d_parent_vertice, d_visited, d_closest);
    }

    cudaEventRecord(e_stop);

    err = cudaGetLastError();

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch gpu kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(distance, d_distance, size, cudaMemcpyDeviceToHost);
    err = cudaMemcpy(parent_vertice, d_parent_vertice, size, cudaMemcpyDeviceToHost);


    // Print execution time
    cudaEventSynchronize(e_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, e_start, e_stop);

    printf("\napp_v2 executed in %f milliseconds", milliseconds);

	// Printing the distance
 	printf("\nVertex \t Distance from Source\n");
    for (int i = 0; i < n; i++){
        printf("%d\t\t%d\n", i, distance[i]);
	}
    	
	//Free gpu memories
    cudaFree(d_graph);
    cudaFree(d_distance);
    cudaFree(d_parent_vertice);
    cudaFree(d_visited);

    //Free cpu memories
    free(distance);
    free(parent_vertice);
    free(visited);
    free(graph);

    return EXIT_SUCCESS;
}