#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

// Number of vertices in the graph
#define MAX pow(2, 28)
#define INF 99999
#define THREADS_BLOCK 1

int n;
int* graph;


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

void read_data(char *file_data){
    
    FILE *file; 
    file = fopen("data.txt", "r");
    
    if ( file == NULL ){
        printf( "data.txt file failed to open.\n" ) ;
    }
    else{

        fgets (file_data, MAX, file);
        fclose(file) ;
    }
}

void init(){

    char  *file_data = (char * )malloc(MAX * sizeof(char));
    read_data(file_data);

    char *parsed_num = strtok(file_data," ");

    int i = 0;

    while (parsed_num != NULL){

        if(i == 0){

            n = atoi(parsed_num);
            graph = (int*) malloc(n * n * sizeof(int));
        }else{

            graph[i-1] = atoi(parsed_num);
        }
        
        i++;
        parsed_num = strtok(NULL, " ");
    }
}

int main(){
	int start = 0;

	init();

	int* distance = (int*) malloc(n * sizeof(int));
	int* pred = (int*) malloc(n * sizeof(int));
	int* visited = (int*) malloc(n * sizeof(int));

	int* cost = (int*) malloc(n * n * sizeof(int));
	
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

	int size = n  * sizeof(int);

	int *d_graph = NULL;
    err = cudaMalloc((void **)&d_graph, n * n * sizeof(int));

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
    err = cudaMalloc((void **)&d_cost, n * n * sizeof(int));

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }


	err = cudaMemcpy(d_graph, graph, n * n * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_distance, distance, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_pred, pred, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_visited, visited, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_cost, cost, n * n * sizeof(int), cudaMemcpyHostToDevice);

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

    printf("\napp v1 executed for %dx%d matrix in %f milliseconds\n", n, n, milliseconds);


	/*
	// Printing the distance
 	printf("\nVertex \t Distance from Source\n");
    for (int i = 0; i < n; i++){
        printf("%d\t\t%d\n", i, distance[i]);
	}
	*/
    	
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