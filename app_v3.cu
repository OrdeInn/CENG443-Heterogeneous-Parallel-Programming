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

__global__ void cudaKernel1(int* graph, int* Ma, int* Ca, int* Ua, int n) {

    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(Ma[tId] == true){

        Ma[tId] = false;


        for(int j = 0; j < n; j++){

            if( graph[j + (tId*n)] != 0 && Ua[tId] < Ca[tId] + graph[j + (tId*n)] ){
                    
                //printf("\nCa:%d\t\tgraph:%d", Ca[tId], graph[j + (i*n)]);
                    
                Ua[tId] = Ca[tId] + graph[j + (tId*n)];
            }
        }
    }
}

__global__ void cudaKernel2(int* Ma, int* Ca, int* Ua, int n) {

    int tId = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("\nCa:%d\t\tUa:%d", Ca[tId], Ua[tId]);

    if(Ca[tId] > Ua[tId]){

        printf("\nCa:%d\t\tUa:%d", Ca[tId], Ua[tId]);

        Ca[tId] = Ua[tId];
        Ma[tId] = true;
    }

    Ua[tId] = Ca[tId];
}

bool isEmpty(int* Ma){

    for(int i = 0; i < n; i++){
        
        if(Ma[i] == false){


            return false;
        }
    }

    return true;
}

void cuda_SSSP(int source){

    int* Ma;
    int* Ca;
    int* Ua;

    Ma = (int*) malloc( n * sizeof(int));
    Ca = (int*) malloc( n * sizeof(int));
    Ua = (int*) malloc( n * sizeof(int));

    for(int i = 0; i < n; i++){

        Ma[i] = false;
        Ca[i] = INF;
        Ua[i] = INF;
    }

    Ma[source] = true;
    Ca[source] = 0;
    Ua[source] = 0;

    for (int i = 0; i < 1; i++){
        for(int j = 0; j < n; j++){
            
            if(graph[j + (i*n)] != 0){
                Ca[j] = graph[j + (i*n)];
            }
        }
    }

    int* d_Ma;
    int* d_Ca;
    int* d_Ua;
    int* d_graph;
	
    cudaError_t err = cudaSuccess;


    err = cudaMalloc((void **)&d_Ma, n * sizeof(int));

    if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_Ca, n * sizeof(int));

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_Ua, n * sizeof(int));

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_graph, n * n * sizeof(int));

	if (err != cudaSuccess){

       	fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
       	exit(EXIT_FAILURE);
    }



    err = cudaMemcpy(d_Ma, Ma, n * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_Ca, Ca, n * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_Ua, Ua, n * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_graph, graph, n * n * sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 grid(9, 1);
    dim3 block(1, 1);

    //Measure execution Time
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    
    cudaEventRecord(e_start);

    while(!isEmpty(Ma)){

        cudaKernel1<<<grid, block>>>(d_graph, d_Ma, d_Ca, d_Ua, n);
        cudaKernel2<<<grid, block>>>(d_Ma, d_Ca, d_Ua, n);

        err = cudaMemcpy(Ma, d_Ma, n * sizeof(int), cudaMemcpyDeviceToHost);
        
        if (err != cudaSuccess){

            fprintf(stderr, "Failed to launch gpu kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
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
    err = cudaMemcpy(Ca, d_Ca, n * sizeof(int), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get data from d_Ca to Ca(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print execution time
    cudaEventSynchronize(e_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, e_start, e_stop);

    printf("\napp_v3 executed for %dx%d matrix in %f milliseconds\n", n, n, milliseconds);



	// Printing the distance
 	printf("\nVertex \t Distance from Source\n");
    for (int i = 0; i < n; i++){

        printf("%d\t\t%d\n", i, Ca[i]);
	}


	//Free gpu memories
    cudaFree(d_Ca);
    cudaFree(d_Ma);
    cudaFree(d_Ua);
    cudaFree(d_graph);

    //Free cpu memories
    free(Ca);
    free(Ma);
    free(Ua);
    free(graph);
}



void read_data(char *file_data){
    
    FILE *file; 
    file = fopen("9x9.txt", "r");
    
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

    free(file_data);
}



int main(){

	int start = 0;

    init();

    cuda_SSSP(start);



    return EXIT_SUCCESS;
}