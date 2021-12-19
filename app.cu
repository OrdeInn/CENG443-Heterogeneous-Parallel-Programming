#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

// Number of vertices in the graph
#define MAX 9
#define INF 99999


__global__ void kernel_dijkstra(int* graph, int* cost, int* distance, int* predecessor, int* visited, int n, int start){


	distance[start] = 0;
	visited[start] = 1;

	int minDist;
	int nextnode;
	

	for(int k = 1; k < n-1; k++){

		minDist = INF;

		for (int i = 0; i < n; i++){
			
			if (distance[i] < minDist && !visited[i]) {
				minDist = distance[i];
				nextnode = i;
			}
		}
		
		visited[nextnode] = 1;

		for (int i = 0; i < n; i++){
			
			if (!visited[i]){

				if (minDist + cost[i + (nextnode*n)] < distance[i]) {  

					distance[i] = minDist + cost[i + (nextnode*n)];
					predecessor[i] = nextnode;
				}
			}
		}

	}		
}

int main(){

	int n = 9;
	int start = 0;

	int* graph = (int*) malloc(MAX * MAX * sizeof(int));
	int* distance = (int*) malloc(MAX * sizeof(int));
	int* pred = (int*) malloc(MAX * sizeof(int));
	int* visited = (int*) malloc(MAX * sizeof(int));

	int* cost = (int*) malloc(MAX * MAX * sizeof(int));

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
	
	 // Creating cost matrix
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		if (graph[j + (i*n)] == 0)
			cost[j + (i*n)] = INF;
		else
			cost[j + (i*n)] = graph[j + (i*n)];

	for (int i = 0; i < n; i++) {
		distance[i] = cost[i + (start*n)]; 
		pred[i] = start;
		visited[i] = 0;
	}

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


	int *d_pred = NULL;
    err = cudaMalloc((void **)&d_pred, size);

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


	int *d_cost = NULL;
    err = cudaMalloc((void **)&d_cost, MAX * MAX * sizeof(int));

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }


	err = cudaMemcpy(d_graph, graph, MAX * MAX * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_distance, distance, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_pred, pred, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_visited, visited, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_cost, cost, MAX * MAX * sizeof(int), cudaMemcpyHostToDevice);

	 if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	dim3 threads(1, 1);
    dim3 blocks(1, 1);


	//Measure execution Time
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    cudaEventRecord(e_start);


	kernel_dijkstra<<<blocks, threads>>>(d_graph, d_cost, d_distance, d_pred, d_visited, n, start);

    err = cudaGetLastError();

	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch gpu kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(e_stop);


	// Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(distance, d_distance, size, cudaMemcpyDeviceToHost);


	// Print execution time
    cudaEventSynchronize(e_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, e_start, e_stop);

    printf("\napp executed in %f milliseconds", milliseconds);



	// Printing the distance
 	printf("\nVertex \t Distance from Source\n");
    for (int i = 0; i < n; i++){
        printf("%d\t\t%d\n", i, distance[i]);
	}
    	
	//Free gpu memories
    cudaFree(d_graph);
    cudaFree(d_distance);
    cudaFree(d_pred);
    cudaFree(d_visited);
    cudaFree(d_cost);

    //Free cpu memories
    free(distance);
    free(pred);
    free(visited);
    free(cost);



    return EXIT_SUCCESS;
}