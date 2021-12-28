#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>


// Number of vertices in the graph
#define MAX pow(2, 28)
#define INF 9999999
#define THREADS_BLOCK 32

int n;
int e;

__global__ void cuda_relax_edge(int n, int* d_V, int* d_I, int* d_E, int* d_W, int* distance, int* temp_distance){

    int tId = threadIdx.x + (blockDim.x * blockIdx.x);
    int step = blockDim.x * gridDim.x;

    for(int i = tId; i < n; i+=step){

       
        for (int j = d_I[i]; j < d_I[i + 1]; j++) {

            int w = d_W[j];

            int du = distance[i];
            int dv = distance[d_E[j]];
            int newDist = du + w;

            if (newDist < dv) {

                temp_distance[d_E[j]] = newDist;
            }
        }
    }
}

__global__ void cuda_update_distance(int n, int* d_V, int* d_I, int* d_E, int* d_W, int* distance, int* temp_distance){


    int tId = threadIdx.x + (blockDim.x * blockIdx.x);
    int step = blockDim.x * gridDim.x;

    for (int i = tId; i < n ; i += step) {  

        if (distance[i] > temp_distance[i]) {

            distance[i] = temp_distance[i];
        }
        temp_distance[i] = distance[i];
    }
}


void read_file(char *file_data, char* file_name){
    
    FILE *file; 
    file = fopen(file_name, "r");
    
    if ( file == NULL ){
        printf( "data.txt file failed to open.\n" ) ;
    }
    else{

        fgets (file_data, MAX, file);
        fclose(file) ;
    }
}


void init(int* V, int*  I, int*  W, int*  E){

    char  *v = (char * )malloc(MAX * sizeof(char));
    char  *i = (char * )malloc(MAX * sizeof(char));
    char  *w = (char * )malloc(MAX * sizeof(char));
    char  *e = (char * )malloc(MAX * sizeof(char));

    char v_file[] = "v2.txt";
    char e_file[] = "e2.txt";
    char w_file[] = "w2.txt";
    char i_file[] = "i2.txt";

    read_file(v, v_file);
    read_file(i, i_file);
    read_file(w, w_file);
    read_file(e, e_file);
    
    
    //Get V
    char *parsed_num = strtok(v," ");
    int index = 0;
    while (parsed_num != NULL){

        V[index] = atoi(parsed_num);
        
        index++;
        parsed_num = strtok(NULL, " ");
    }

    //Get W
    parsed_num = strtok(w," ");
    index = 0;
    while (parsed_num != NULL){

    
        W[index] = atoi(parsed_num);
       
        
        index++;
        parsed_num = strtok(NULL, " ");
    }

    //Get I
    parsed_num = strtok(i," ");
    index = 0;
    while (parsed_num != NULL){

    
        I[index] = atoi(parsed_num);
       
        
        index++;
        parsed_num = strtok(NULL, " ");
    }


    //Get E
    parsed_num = strtok(e," ");    
    index = 0;
    while (parsed_num != NULL){

        
        E[index] = atoi(parsed_num);
        
        
        index++;
        parsed_num = strtok(NULL, " ");
    }
}

int main(){

	int start = 0;

    n = 435666; // Number of vertices
    e = 1057066;   //Number of edges

    int* V;
    int* I;
    int* E;
    int* W;
    int* distance;
    int* temp_distance;

    V = (int*) malloc( n * sizeof(int));
    I = (int*) malloc( n * sizeof(int));
    W = (int*) malloc( e * sizeof(int));
    E = (int*) malloc( e * sizeof(int));
    distance = (int*) malloc( n * sizeof(int));
    temp_distance = (int*) malloc( n * sizeof(int));

    init(V, I, W, E);
	
    for(int i = 0; i < n; i++){

        if(i == start){

            distance[i] = 0;
            temp_distance[i] = 0;
        }else{

            distance[i] = INF;
            temp_distance[i] = INF;
        }
    }

    int* d_V;
    int* d_W;
    int* d_E;
    int* d_I;
    int* d_distance;
    int* d_temp_distance;

    cudaMalloc((void **)&d_V, n * sizeof(int));
    cudaMalloc((void **)&d_W, e * sizeof(int));
    cudaMalloc((void **)&d_E, e * sizeof(int));
    cudaMalloc((void **)&d_I, n * sizeof(int));
    cudaMalloc((void **)&d_distance, n * sizeof(int));
    cudaMalloc((void **)&d_temp_distance, n * sizeof(int));

    cudaMemcpy(d_V, V, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, e * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, e * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, I, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, distance, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_distance, temp_distance, n * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threads(1024, 1);
    dim3 blocks(32, 1);


    //Measure execution Time
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    cudaEventRecord(e_start);

    for(int i = 0; i < n; i++){

        cuda_relax_edge<<<blocks, threads>>>(n, d_V, d_I, d_E, d_W, d_distance, d_temp_distance);
        cuda_update_distance<<<blocks, threads>>>(n, d_V, d_I, d_E, d_W, d_distance, d_temp_distance);
    }

    cudaEventRecord(e_stop);


    cudaMemcpy(distance, d_distance, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print execution time
    cudaEventSynchronize(e_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, e_start, e_stop);

    printf("\nBellman-Ford algorithm executed in parallel for a graph with %d vertices and %d edges in %f milliseconds\n", n, e, milliseconds);

    /*
	// Printing the distance
 	printf("\nVertex \t Distance from Source\n");
    for (int i = 0; i < n; i++){
        printf("%d\t\t%d\n", i, distance[i]);
	}

    */


    cudaFree(d_V);
    cudaFree(d_W);
    cudaFree(d_I);
    cudaFree(d_E);
    cudaFree(d_distance);
    cudaFree(d_temp_distance);

    free(V);
    free(W);
    free(I);
    free(E);
    free(distance);
    free(temp_distance);

    return EXIT_SUCCESS;
}