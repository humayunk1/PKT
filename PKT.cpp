/* ********************************************************************************************
 *  PKT: Shared-memory Graph Truss Decomposition
 *  Authors: Humayun Kabir and Kamesh Madduri  {hzk134, madduri}@cse.psu.edu 
 *  Description: A parallel graph k-truss decomposition algorithm for shared-memory system. It 
 *  is designed for large sparse graphs.     
 * 
 *  Please cite the following paper, if you are using PKT: 
 *  H. Kabir and K. Madduri, "Shared-memory Graph Truss Decomposition", arXiv.org e-Print archive,
 *  https://arxiv.org/abs/1707.02000, July 2017
 * ********************************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h> 

#include <functional>
#include <iostream>
#include <unordered_map>
#include <climits>

using namespace std;

//Global variables to store num of threads 
int NUM_THREADS = 1;

typedef unsigned int vid_t;
typedef unsigned int eid_t;

typedef struct {
    long n;
    long m;

    vid_t *adj;
    eid_t *num_edges;
    eid_t *eid; 
} graph_t;

void free_graph(graph_t *g) {
    if( g->adj != NULL )
	free( g->adj );

    if( g->num_edges != NULL )
	free( g->num_edges );

    if( g->eid != NULL )
	free( g->eid );
}


//Define an Edge data type
struct Edge {
    vid_t u;
    vid_t v;

    Edge() {
	this->u = 0;
        this->v = 0;
    }

    Edge(vid_t u, vid_t v) {
	this->u = u;
        this->v = v;
    }
};

//Define a hash function for Edge
struct EdgeHasher
{
    std::size_t operator()(const Edge& k) const
    {
	std::hash<vid_t> unsignedHash;
        return ( ( unsignedHash(k.u) >> 1 ) ^ ( unsignedHash(k.v) << 1 ) );
    }
};

struct edgeEqual {
    bool operator()(const Edge& e1, const Edge& e2) const {
	return (e1.u == e2.u) && (e1.v == e2.v);
    }
};

typedef unordered_map<Edge, long, EdgeHasher, edgeEqual> MapType;

static double timer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) (tp.tv_sec) + tp.tv_usec * 1e-6);
}

void read_env() {

#pragma omp parallel
    {
#pragma omp master
	NUM_THREADS = omp_get_num_threads();
    }

    printf("NUM_PROCS:     %d \n", omp_get_num_procs());
    printf("NUM_THREADS:   %d \n", NUM_THREADS);
}

/*********************** READ INPUT FILE  ************************************************************/
int vid_compare (const void * a, const void * b) {   
    return ( *(vid_t*)a - *(vid_t*)b ); 
}


int load_graph_from_file(char *filename, graph_t *g) {

    FILE *infp = fopen(filename, "r");
    if (infp == NULL) {
        fprintf(stderr, "Error: could not open inputh file: %s.\n Exiting ...\n", filename);
        exit(1);
    }

    fprintf(stdout, "Reading input file: %s\n", filename);

    double t0 = timer();

    //Read N and M
    fscanf(infp, "%ld %ld\n", &(g->n), &(g->m));
    printf("N: %ld, M: %ld \n", g->n, g->m);

    long m = 0;

    //Allocate space
    g->num_edges = (eid_t *) malloc((g->n + 1) * sizeof(eid_t));
    assert(g->num_edges != NULL);

#pragma omp parallel for 
    for (long i=0; i<g->n + 1; i++) {
        g->num_edges[i] = 0;
    }

    vid_t u, v;
    while( fscanf(infp, "%u %u\n", &u, &v) != EOF ) {
        m++;
        g->num_edges[u]++;
        g->num_edges[v]++;
    }

    fclose( infp );

    if( m != g->m) {
        printf("Reading error: file does not contain %ld edges.\n", g->m);
        free( g->num_edges );
        exit(1);
    }

    m = 0;

    eid_t *temp_num_edges = (eid_t *) malloc((g->n + 1) * sizeof(eid_t));
    assert(temp_num_edges != NULL);

    temp_num_edges[0] = 0;

    for(long i = 0; i < g->n; i++) {
        m += g->num_edges[i];
        temp_num_edges[i+1] = m;
    }

    //g->m is twice number of edges
    g->m = m;

    //Allocate space for adj
    g->adj = (vid_t *) malloc(m * sizeof(vid_t));
    assert(g->adj != NULL);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(long i = 0; i < g->n+1; i++)
            g->num_edges[i] = temp_num_edges[i];

#pragma omp for schedule(static)
        for(long i = 0; i < m; i++)
            g->adj[i] = 0;
    }


    infp = fopen(filename, "r");
    if (infp == NULL) {
        fprintf(stderr, "Error: could not open input file: %s.\n Exiting ...\n", filename);
        exit(1);
    }

    //Read N and M
    fscanf(infp, "%ld %ld\n", &(g->n), &m);

    //Read the edges
    while( fscanf(infp, "%u %u\n", &u, &v) != EOF ) {
        g->adj[ temp_num_edges[u]  ] = v;
        temp_num_edges[u]++;
        g->adj[ temp_num_edges[v] ] = u;
        temp_num_edges[v]++;
    }

    fclose( infp );

    //Sort the adjacency lists
    for(long i = 0; i < g->n; i++) {
        qsort(g->adj+g->num_edges[i], g->num_edges[i+1]  - g->num_edges[i], sizeof(vid_t), vid_compare);
    }

    fprintf(stdout, "Reading input file took time: %.2lf sec \n", timer() - t0);
    free( temp_num_edges );
    return 0;
}
   

//Populate eid and edge list
void getEidAndEdgeList(graph_t *g, Edge* idToEdge) {

    //Allocate space for eid -- size g->m
    g->eid = (eid_t *)malloc( g->m * sizeof(eid_t) );
    assert( g->eid  != NULL );

    //Edge start of each edge
    eid_t *num_edges_copy = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( num_edges_copy != NULL );

    for(vid_t i = 0; i < g->n; i++)  {
        num_edges_copy[i] = g->num_edges[i];
    }

    long edgeId = 0;

    //Number the edges as <u,v> -- such that u < v -- <u,v> and <v,u> are same edge    
    for(vid_t u = 0; u < g->n; u++) {
        //now go through the adjacencies of u
        for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
            vid_t v = g->adj[j];
            if( u < v ) {
                Edge e;
		e.u = u;
		e.v = v;

		g->eid[j] = edgeId;
		num_edges_copy[u]++;

		if( g->adj[ num_edges_copy[v] ] == u ) {
		    g->eid[ num_edges_copy[v] ] = edgeId;
		    num_edges_copy[v]++;
		}

                idToEdge[ edgeId ] = e;
                edgeId ++;
            }
        }
    }

}


/*******************************************************************************************************/
void PKT_scan(long numEdges, int *EdgeSupport, int level, eid_t *curr, long *currTail, bool *InCurr) {
    // Size of cache line
    const long BUFFER_SIZE_BYTES = 2048;
    const long BUFFER_SIZE = BUFFER_SIZE_BYTES/sizeof(vid_t);

    vid_t buff[BUFFER_SIZE];
    long index = 0;

#pragma omp for schedule(static) 
    for(long i = 0; i < numEdges; i++) {
	if( EdgeSupport[i] == level ) {
	    buff[index] = i;
	    InCurr[i] = true;
	    index ++;

	    if(index >= BUFFER_SIZE) {
		long tempIdx = __sync_fetch_and_add(currTail, BUFFER_SIZE);

		for(long j = 0; j < BUFFER_SIZE; j++) {
		    curr[tempIdx+j] = buff[j];
		}
		index = 0;
	    } 
	}
    }

    if(index > 0) {
	long tempIdx = __sync_fetch_and_add(currTail, index);

	for(long j = 0; j < index; j++) {
	    curr[tempIdx+j] = buff[j];
	}
    }

#pragma omp barrier

}


//Process a sublevel in a level using intersection based approach
void PKT_processSubLevel_intersection(graph_t *g, eid_t *curr, bool *InCurr, long currTail, int *EdgeSupport, 
    int level, eid_t *next, bool *InNext, long *nextTail, bool *processed, Edge * edgeIdtoEdge) {

    //Size of cache line
    const long BUFFER_SIZE_BYTES = 2048;
    const long BUFFER_SIZE = BUFFER_SIZE_BYTES/sizeof(vid_t);

    vid_t buff[BUFFER_SIZE];
    long index = 0;

#pragma omp for schedule(dynamic,4)
    for (long i = 0; i < currTail; i++) {

	//process edge <u,v>
        eid_t e1 = curr[i]; 

	Edge edge = edgeIdtoEdge[e1];  

	vid_t u = edge.u;
	vid_t v = edge.v;

	eid_t uStart = g->num_edges[u], uEnd = g->num_edges[u+1];
        eid_t vStart = g->num_edges[v], vEnd = g->num_edges[v+1];

        unsigned int numElements = (uEnd - uStart) + (vEnd - vStart);
        eid_t j_index = uStart, k_index = vStart;

	for(unsigned int innerIdx = 0; innerIdx < numElements; innerIdx ++) {
	    if( j_index >= uEnd) {
		break;
	    }
            else if( k_index >= vEnd ) {
		break;
            }
            else if( g->adj[j_index] == g->adj[k_index] ) {

		eid_t e2 = g->eid[ k_index ];  //<v,w>
                eid_t e3 = g->eid[ j_index ];  //<u,w>


                //If e1, e2, e3 forms a triangle
		if( (!processed[e2]) && (!processed[e3]) ) {

		    //Decrease support of both e2 and e3
		    if( EdgeSupport[e2] > level && EdgeSupport[e3] > level) {

			//Process e2
			int supE2 = __sync_fetch_and_sub( &EdgeSupport[e2], 1);
			if( supE2 == (level+1) ) {
			    buff[index] = e2;
			    InNext[e2] = true; 
			    index ++;
			}

			if( supE2 <= level ) {
			    __sync_fetch_and_add(&EdgeSupport[e2],1);
			}

			if( index >= BUFFER_SIZE) { 
                            long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);
                        
                            for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                next [tempIdx + bufIdx] = buff[bufIdx];
                            index = 0;			
			}

			//Process e3
			int supE3 = __sync_fetch_and_sub(&EdgeSupport[e3], 1);

			if( supE3 == (level +1) ) {
			    buff[index] = e3;
			    InNext[e3] = true; 
			    index++;
			}

			if(supE3 <= level ) {
			    __sync_fetch_and_add(&EdgeSupport[e3],1);
			}

			if( index >= BUFFER_SIZE) {
			    long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);

                            for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                next [tempIdx + bufIdx] = buff[bufIdx];
                            index = 0; 
                        }				

		    }
		    else if(EdgeSupport[e2] > level ) {

			//process e2 only if e1 < e3
			if( e1 < e3 && InCurr[e3] ) {
			    int supE2 = __sync_fetch_and_sub(&EdgeSupport[e2], 1);

                            if( supE2 == (level+1) ) {
                                buff[index] = e2;
			        InNext[e2] = true; 
                                index ++;
                            }

                            if( supE2 <= level ) {
                                __sync_fetch_and_add(&EdgeSupport[e2],1);
                            }
				
                            if( index >= BUFFER_SIZE) {
                                long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);

                                for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                    next [tempIdx + bufIdx] = buff[bufIdx];
                                index = 0;
                            }						
			}
			if( !InCurr[e3] ) { //if e3 is not in curr array then decrease support of e2
                            int supE2 = __sync_fetch_and_sub(&EdgeSupport[e2], 1);                                  
                            if( supE2 == (level+1) ) {
                                buff[index] = e2;
			        InNext[e2] = true; 
                                index ++;
                            }
                    
                            if( supE2 <= level ) {
                                __sync_fetch_and_add(&EdgeSupport[e2],1);
                            }
                                 
                            if( index >= BUFFER_SIZE) {
                                long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);

                                for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                    next [tempIdx + bufIdx] = buff[bufIdx];
                                index = 0;
                            }			
			}
		    }
		    else if(EdgeSupport[e3] > level) {

			//process e3 only if e1 < e2
			if( e1 < e2 && InCurr[e2] ) {
                            int supE3 = __sync_fetch_and_sub(&EdgeSupport[e3], 1);

                            if( supE3 == (level +1) ) {
                                buff[index] = e3;
			        InNext[e3] = true; 
                                index++;
                            }

                            if(supE3 <= level ) {
                                __sync_fetch_and_add(&EdgeSupport[e3],1);
                            }

                            if( index >= BUFFER_SIZE) {
                                long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);

                                for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                    next [tempIdx + bufIdx] = buff[bufIdx];
                                index = 0;
                            }
			}
			if( !InCurr[e2] ) { //if e2 is not in curr array then decrease support of e3 
                            int supE3 = __sync_fetch_and_sub(&EdgeSupport[e3], 1);
                            
                            if( supE3 == (level +1) ) {
                                buff[index] = e3;
			        InNext[e3] = true; 
                                index++;
                            }
                            
                            if(supE3 <= level ) {
                                __sync_fetch_and_add(&EdgeSupport[e3],1);
                            }
                            
                            if( index >= BUFFER_SIZE) {
                                long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);
                                
                                for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                    next [tempIdx + bufIdx] = buff[bufIdx];
                                index = 0;
                            }			

			}
		    } 

		}	


                j_index ++;
                k_index ++;
            }
            else if( g->adj[j_index] < g->adj[k_index] ) {
                j_index++;
            }
            else if( g->adj[k_index] < g->adj[j_index] ) {
                k_index++;
            }
	}
    }


    if (index > 0) {
        long tempIdx =  __sync_fetch_and_add(nextTail, index);;
        for (long bufIdx = 0; bufIdx < index; bufIdx++)
            next [tempIdx + bufIdx] = buff[bufIdx];
    }

#pragma omp barrier

#pragma omp for schedule(static)
    for (long i = 0; i < currTail; i++) {
        eid_t e = curr[i];  

	processed[e] = true;
	InCurr[e] = false;
    }

#pragma omp barrier


}


//Process a sublevel in a level using marking based approach
void PKT_processSubLevel_marking(graph_t *g, eid_t *curr, bool *InCurr, long currTail, int *EdgeSupport, int level, 
    eid_t *next, bool *InNext, long *nextTail, eid_t *X, bool *processed, Edge * edgeIdtoEdge) {

    //Size of cache line
    const long BUFFER_SIZE_BYTES = 2048;
    const long BUFFER_SIZE = BUFFER_SIZE_BYTES/sizeof(vid_t);

    vid_t buff[BUFFER_SIZE];
    long index = 0;

#pragma omp for schedule(dynamic,4)
    for (long i = 0; i < currTail; i++) {

	//process edge <u,v> 
        eid_t e1 = curr[i];

	Edge edge = edgeIdtoEdge[e1]; 

	vid_t u = edge.u;
	vid_t v = edge.v;

	for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
	    vid_t w = g->adj[j];
	    if( w != v) 
		X[w] = j+1;  
	}

        //Check the adj list of vertex v
        for (eid_t j = g->num_edges[v]; j < g->num_edges[v+1]; j++) {
	    vid_t w = g->adj[j];
	    if( X[w] ) {
		eid_t e2 = g->eid[j];  //<v,w>
		eid_t e3 = g->eid[ X[w] -1]; //<u,w> 
	
		//If e1, e2, e3 forms a triangle
		if( (!processed[e2]) && (!processed[e3]) ) {

		    //Decrease support of both e2 and e3
		    if( EdgeSupport[e2] > level && EdgeSupport[e3] > level) {

			//Process e2
			int supE2 = __sync_fetch_and_sub( &EdgeSupport[e2], 1);
			if( supE2 == (level+1) ) {
			    buff[index] = e2;
			    InNext[e2] = true; 
			    index ++;
			}

			if( supE2 <= level ) {
			    __sync_fetch_and_add(&EdgeSupport[e2],1);
			}

			if( index >= BUFFER_SIZE) { 
                            long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);
                        
                            for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                next [tempIdx + bufIdx] = buff[bufIdx];
                            index = 0;			
			}

			//Process e3
			int supE3 = __sync_fetch_and_sub(&EdgeSupport[e3], 1);

			if( supE3 == (level +1) ) {
			    buff[index] = e3;
			    InNext[e3] = true; 
			    index++;
			}

			if(supE3 <= level ) {
			    __sync_fetch_and_add(&EdgeSupport[e3],1);
			}

			if( index >= BUFFER_SIZE) {
			    long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);

                            for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                next [tempIdx + bufIdx] = buff[bufIdx];
                            index = 0; 
                        }				

		    }
		    else if(EdgeSupport[e2] > level ) {

			//process e2 only if e1 < e3
			if( e1 < e3 && InCurr[e3] ) {
			    int supE2 = __sync_fetch_and_sub(&EdgeSupport[e2], 1);

                            if( supE2 == (level+1) ) {
                                buff[index] = e2;
			        InNext[e2] = true; 
                                index ++;
                            }

                            if( supE2 <= level ) {
                                __sync_fetch_and_add(&EdgeSupport[e2],1);
                            }
				
                            if( index >= BUFFER_SIZE) {
                                long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);

                                for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                    next [tempIdx + bufIdx] = buff[bufIdx];
                                index = 0;
                            }						
			}
			if( !InCurr[e3] ) { //if e3 is not in curr array then decrease support of e2
                            int supE2 = __sync_fetch_and_sub(&EdgeSupport[e2], 1);                                  
                            if( supE2 == (level+1) ) {
                                buff[index] = e2;
			        InNext[e2] = true; 
                                index ++;
                            }
                    
                            if( supE2 <= level ) {
                                __sync_fetch_and_add(&EdgeSupport[e2],1);
                            }
                                 
                            if( index >= BUFFER_SIZE) {
                                long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);

                                for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                    next [tempIdx + bufIdx] = buff[bufIdx];
                                index = 0;
                            }			
			}
		    }
		    else if(EdgeSupport[e3] > level) {
			//process e3 only if e1 < e2
			if( e1 < e2 && InCurr[e2] ) {
                            int supE3 = __sync_fetch_and_sub(&EdgeSupport[e3], 1);

                            if( supE3 == (level +1) ) {
                                buff[index] = e3;
			        InNext[e3] = true; 
                                index++;
                            }

                            if(supE3 <= level ) {
                                __sync_fetch_and_add(&EdgeSupport[e3],1);
                            }

                            if( index >= BUFFER_SIZE) {
                                long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);

                                for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                    next [tempIdx + bufIdx] = buff[bufIdx];
                                index = 0;
                            }
			}
			if( !InCurr[e2] ) { //if e2 is not in curr array then decrease support of e3 
                            int supE3 = __sync_fetch_and_sub(&EdgeSupport[e3], 1);
                            
                            if( supE3 == (level +1) ) {
                                buff[index] = e3;
			        InNext[e3] = true; 
                                index++;
                            }
                            
                            if(supE3 <= level ) {
                                __sync_fetch_and_add(&EdgeSupport[e3],1);
                            }
                            
                            if( index >= BUFFER_SIZE) {
                                long tempIdx = __sync_fetch_and_add(nextTail, BUFFER_SIZE);
                                
                                for(long bufIdx = 0; bufIdx < BUFFER_SIZE; bufIdx++)
                                    next [tempIdx + bufIdx] = buff[bufIdx];
                                index = 0;
                            }			

			}
		    } 

		}	

	    }

        }

	//Unmark X 
	for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
	    vid_t w = g->adj[j];
	    if( w != v) 
		X[w] = 0;  
	}
	
    }


    if (index > 0) {
        long tempIdx =  __sync_fetch_and_add(nextTail, index);;
        for (long bufIdx = 0; bufIdx < index; bufIdx++)
            next [tempIdx + bufIdx] = buff[bufIdx];
    }

#pragma omp barrier

#pragma omp for schedule(static)
    for (long i = 0; i < currTail; i++) {
        eid_t e = curr[i]; 

	processed[e] = true;
	InCurr[e] = false;
    }

#pragma omp barrier


}



/**   Computes the support of each edge in parallel 
 *    Computes k-truss in parallel   ****/
void PKT_intersection(graph_t *g, int *EdgeSupport, Edge *edgeIdToEdge) {

    long numEdges = g->m / 2;
    long n = g->n;

    bool *processed = (bool *)malloc( numEdges * sizeof(bool) );
    assert( processed != NULL );

    long currTail = 0;   
    long nextTail = 0;   

    eid_t *curr = (eid_t *)malloc( numEdges * sizeof(eid_t) ); 
    assert( curr != NULL ); 

    bool *InCurr = (bool *)malloc( numEdges * sizeof(bool) ); 
    assert( InCurr != NULL ); 

    eid_t *next = (eid_t *)malloc( numEdges * sizeof(eid_t) ); 
    assert( next != NULL );

    bool *InNext = (bool *)malloc( numEdges * sizeof(bool) ); 
    assert( InNext != NULL ); 

    eid_t *startEdge = (eid_t *)malloc(n * sizeof(eid_t) );
    assert( startEdge != NULL );

 
    //parallel region
#pragma omp parallel 
{ 
    int tid = omp_get_thread_num();

    eid_t *X = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( X != NULL );

    for(vid_t i = 0; i < g->n; i++) {
	X[i] = 0;
    }

#pragma omp for schedule(static) 
    for( eid_t e = 0; e < numEdges; e++ ) {
	//Initialize processed array with false
	processed[ e ] = false;
	
	InCurr[e] = false;
	InNext[e] = false;
    }



#pragma omp for schedule(static) 
    for( vid_t i = 0; i < n; i++ ) {
	eid_t j = g->num_edges[i];
	eid_t endIndex = g->num_edges[i+1];

	while( j < endIndex ) {
	    if(g->adj[j] > i)
		break;
	    j++;
	}
	startEdge[i] = j;
    } 

#if TIME_RESULTS
    double triTime = 0;
    double scanTime = 0;
    double procTime = 0;
    double start =  timer();
#endif

#pragma omp for schedule(dynamic,10) 
    for( vid_t u = 0; u < n; u++ ) {

        for(eid_t j = startEdge[u]; j < g->num_edges[u+1]; j++) {
	    vid_t w = g->adj[j];
	    X[w] = j +1;
	}

        for(eid_t j = g->num_edges[u]; j < startEdge[u]; j++) {
	    vid_t v = g->adj[j];

	    for(eid_t k = g->num_edges[v+1]-1; k >= startEdge[v]; k --) {
                vid_t w = g->adj[k];
                // check if: w > u
                if( w <= u ) {
		    break;
		}

                if(  X[w] ) {  //This is a triangle
                    //edge id's are: <u,w> : g->eid[ X[w] -1] 
                    //<u,w> : g->eid[ X[w] -1] 
                    //<v,u> : g->eid[ j ]  
                    //<v,w> : g->eid[ k ]		
                    eid_t e1 = g->eid[ X[w] -1 ] , e2 = g->eid[j], e3 = g->eid[k];
                    __sync_fetch_and_add(&EdgeSupport[e1], 1);
                    __sync_fetch_and_add(&EdgeSupport[e2], 1);
                    __sync_fetch_and_add(&EdgeSupport[e3], 1);
                } 
	    }                   
	}
	
	for(eid_t j = startEdge[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            X[w] = 0;
        }
    }


#pragma omp barrier

#if TIME_RESULTS
    triTime = timer() - start ;
    start = timer();
#endif

    //Support computation is done
    //Computing truss now

    int level = 0;
    long todo = numEdges;
   
    while( todo > 0 ) {

#if TIME_RESULTS
	start = timer();
#endif

	PKT_scan(numEdges, EdgeSupport, level, curr, &currTail, InCurr);

#if TIME_RESULTS
	scanTime += timer() - start;
	start = timer();
#endif       

	while( currTail > 0 ) {
	    todo = todo - currTail;
	
	    PKT_processSubLevel_intersection(g, curr, InCurr, currTail, EdgeSupport, level, next, InNext, &nextTail, processed, edgeIdToEdge);

	    if( tid == 0 ) {
		eid_t *tempCurr = curr;
		curr = next;
		next = tempCurr;

		bool *tempInCurr = InCurr;
		InCurr = InNext;
		InNext = tempInCurr;


		currTail = nextTail;
		nextTail = 0;
	    }

#pragma omp barrier	
	}

#if TIME_RESULTS
	procTime += timer() - start;
#endif 

	level = level + 1;
#pragma omp barrier	

    }

#if TIME_RESULTS
    if(tid == 0) {
        printf("Tri time: %9.3lf \nScan Time: %9.3lf \nProc Time: %9.3lf \n", triTime, scanTime, procTime);
        printf("PKT-intersection-Time: %9.3lf\n", triTime + scanTime + procTime);
    }
#endif

    free( X );

}  //End of parallel region

    //Free memory
    free( next );
    free( InNext );
    free( curr );
    free( InCurr );
    free( processed );
    free( startEdge ); 
}


void PKT_scan_serial(long numEdges, int *EdgeSupport, int level, eid_t *curr, long *currTail) {
    for(long i = 0; i < numEdges; i++) {
        if( EdgeSupport[i] == level ) {
            curr[ (*currTail) ] = i;
            (*currTail) = (*currTail) + 1;
        }
    }
}


//Serially process sublevel in a level using intersection
void PKT_processSubLevel_serial_intersection(graph_t *g, eid_t *curr, long currTail, int *EdgeSupport, 
    int level, eid_t *next, long *nextTail, bool *processed, Edge * edgeIdtoEdge) {

    for (long i = 0; i < currTail; i++) {
	
	//process edge <u,v>
        eid_t e1 = curr[i];

        Edge edge = edgeIdtoEdge[e1];   

        vid_t u = edge.u;
        vid_t v = edge.v;


        eid_t uStart = g->num_edges[u], uEnd = g->num_edges[u+1];
        eid_t vStart = g->num_edges[v], vEnd = g->num_edges[v+1];

        unsigned int numElements = (uEnd - uStart) + (vEnd - vStart);
        eid_t j_index = uStart, k_index = vStart;

        for(unsigned int innerIdx = 0; innerIdx < numElements; innerIdx ++) {
            if( j_index >= uEnd) {
                break;
            }
            else if( k_index >= vEnd ) {
                break;
            }
            else if( g->adj[j_index] == g->adj[k_index] ) {

                eid_t e2 = g->eid[k_index];  //<v,w>
                eid_t e3 = g->eid[ j_index ]; //<u,w> 

                //If e1, e2, e3 forms a triangle
                if( (!processed[e2]) && (!processed[e3]) ) {

                    //Decrease support of both e2 and e3           
                    if( EdgeSupport[e2] > level && EdgeSupport[e3] > level) {

                        //Process e2            
                        EdgeSupport[e2] = EdgeSupport[e2] -1;
                        if( EdgeSupport[e2] == level ) {
                            next [ (*nextTail) ] = e2;
                            (*nextTail) = (*nextTail) + 1;
                        }

                        //Process e3             
                        EdgeSupport[e3] = EdgeSupport[e3] - 1;

                        if( EdgeSupport[e3]  == level ) {
                            next [ (*nextTail) ] = e3;
                            (*nextTail) = (*nextTail) + 1;
                        }
                    }
                    else if(EdgeSupport[e2] > level ) {              

                        //process e2 
                        EdgeSupport[e2] = EdgeSupport[e2] -1;

                        if( EdgeSupport[e2] == level ) {
                            next [ (*nextTail) ] = e2;
                            (*nextTail) = (*nextTail) + 1;
                        }
                    }
                    else if(EdgeSupport[e3] > level) {
                        //process e3 
                        EdgeSupport[e3] = EdgeSupport[e3]  - 1;

                        if( EdgeSupport[e3] == level ) {
                            next [ (*nextTail) ] = e3;
                            (*nextTail) = (*nextTail) + 1;
                        }
                    }
                }


                j_index ++;
                k_index ++;
            }
            else if( g->adj[j_index] < g->adj[k_index] ) {
                j_index++;
            }
            else if( g->adj[k_index] < g->adj[j_index] ) {
                k_index++;
            }
        }
	processed[e1] = true;
    }
}

//Serially process sublevel in a level using marking
void PKT_processSubLevel_serial_marking(graph_t *g, eid_t *curr, long currTail, int *EdgeSupport, 
    int level, eid_t *next, long *nextTail, eid_t *X, bool *processed, Edge * edgeIdtoEdge) {

    for (long i = 0; i < currTail; i++) {

	//process edge <u,v>
        eid_t e1 = curr[i]; 

        Edge edge = edgeIdtoEdge[e1];   

        vid_t u = edge.u;
        vid_t v = edge.v;

        for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            if( w != v)
                X[w] = j+1;  
        }

        //Check the adj list of vertex v
        for (eid_t j = g->num_edges[v]; j < g->num_edges[v+1]; j++) {
            vid_t w = g->adj[j];

            if( X[w] ) {
                eid_t e2 = g->eid[j];  //<v,w>
                eid_t e3 = g->eid[ X[w] -1]; //<u,w> 

                //If e1, e2, e3 forms a triangle
                if( (!processed[e2]) && (!processed[e3]) ) {

                    //Decrease support of both e2 and e3
                    if( EdgeSupport[e2] > level && EdgeSupport[e3] > level) {

                        //Process e2
                        EdgeSupport[e2] = EdgeSupport[e2] -1;
                        if( EdgeSupport[e2] == level ) {
                            next [ (*nextTail) ] = e2;
                            (*nextTail) = (*nextTail) + 1;
                        }

                        //Process e3
                        EdgeSupport[e3] = EdgeSupport[e3] - 1;

                        if( EdgeSupport[e3]  == level ) {
                            next [ (*nextTail) ] = e3;
                            (*nextTail) = (*nextTail) + 1;
                        }
                    }
                    else if(EdgeSupport[e2] > level ) {
                        //process e2 
                        EdgeSupport[e2] = EdgeSupport[e2] -1;

                        if( EdgeSupport[e2] == level ) {
                            next [ (*nextTail) ] = e2;
                            (*nextTail) = (*nextTail) + 1;
                        }
                    }
                    else if(EdgeSupport[e3] > level) {
                        //process e3 
                        EdgeSupport[e3] = EdgeSupport[e3]  - 1;

                        if( EdgeSupport[e3] == level ) {
                            next [ (*nextTail) ] = e3;
                            (*nextTail) = (*nextTail) + 1;
                        }
                    }
                }

            }

        }

        //Unmark X 
        for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            if( w != v)
                X[w] = 0;  
        }

	processed[e1] = true;

    }

}
                    
                     
                      
/**   Serial PKT_intersection Algorithm  ***/
void PKT_serial_intersection(graph_t *g, int *EdgeSupport, Edge *edgeIdToEdge) {

    long numEdges = g->m / 2;
    long n = g->n;

    //An array to mark processed array
    bool *processed = (bool *)malloc( numEdges * sizeof(bool) );
    assert( processed != NULL );

    long currTail = 0;
    long nextTail = 0;

    eid_t *curr = (eid_t *)malloc( numEdges * sizeof(eid_t) );
    assert( curr != NULL );

    eid_t *next = (eid_t *)malloc( numEdges * sizeof(eid_t) );
    assert( next != NULL );

    eid_t *startEdge = (eid_t *)malloc(n * sizeof(eid_t) );
    assert( startEdge != NULL );


    eid_t *X = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( X != NULL );

    for(vid_t i = 0; i < g->n; i++) {
        X[i] = 0;
    }

    //Initialize the arrays
    for( eid_t e = 0; e < numEdges; e++ ) {
        processed[ e ] = false;
    }

    //Find the startEdge for each vertex
    for( vid_t i = 0; i < n; i++ ) {
        eid_t j = g->num_edges[i];
        eid_t endIndex = g->num_edges[i+1];

        while( j < endIndex ) {
            if(g->adj[j] > i)
                break;
            j++;
        }
        startEdge[i] = j;
    }

#if TIME_RESULTS
    double triTime = 0;
    double scanTime = 0;
    double procTime = 0;
    double startTime = timer();
#endif

    for( vid_t u = 0; u < n; u++ ) {

        for(eid_t j = startEdge[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            X[w] = j +1;
        }

        for(eid_t j = g->num_edges[u]; j < startEdge[u]; j++) {
            vid_t v = g->adj[j];

            for(eid_t k = g->num_edges[v+1]-1; k >= startEdge[v]; k --) {
                vid_t w = g->adj[k];
                // check if: w > u
                if( w <= u ) {
                    break;
                }

                if(  X[w] ) {  //This is a triangle
                    //edge id's are: <u,w> : g->eid[ X[w] -1] 
                    //<u,w> : g->eid[ X[w] -1] 
                    //<v,u> : g->eid[ j ]  
                    //<v,w> : g->eid[ k ]   
                    eid_t e1 = g->eid[ X[w] -1 ] , e2 = g->eid[j], e3 = g->eid[k];
                    EdgeSupport[e1] += 1;
                    EdgeSupport[e2] += 1;
                    EdgeSupport[e3] += 1;
                }
            }
        }

        for(eid_t j = startEdge[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            X[w] = 0;
        }
    }

#if TIME_RESULTS
    triTime = timer() - startTime ;
    startTime = timer();
#endif

    //Support computation is done
    //Computing support now

    int level = 0;
    long todo = numEdges;

    while( todo > 0 ) {

#if TIME_RESULTS
        startTime = timer();
#endif

        PKT_scan_serial(numEdges, EdgeSupport, level, curr, &currTail);

#if TIME_RESULTS
        scanTime += timer() - startTime;
        startTime = timer();
#endif

        while( currTail > 0 ) {
            todo = todo - currTail;

            PKT_processSubLevel_serial_intersection(g, curr, currTail, EdgeSupport, level, next, &nextTail, processed, edgeIdToEdge);

            eid_t *tempCurr = curr;
            curr = next;
            next = tempCurr;

            currTail = nextTail;
            nextTail = 0;
        }

#if TIME_RESULTS
        procTime += timer() - startTime;
#endif
        level = level + 1;
    }

#if TIME_RESULTS
    printf("Tri time: %9.3lf \nScan Time: %9.3lf \nProc Time: %9.3lf \n", triTime, scanTime, procTime);
    printf("PKT-serial-Time-Intersection: %9.3lf\n", triTime + scanTime + procTime);
#endif


    free( X );

    //Free memory
    free( next );
    free( curr );
    free( processed );
    free( startEdge );
}

/**   Serial PKT_marking Algorithm  ***/
void PKT_serial_marking(graph_t *g, int *EdgeSupport, Edge *edgeIdToEdge) {

    long numEdges = g->m / 2;
    long n = g->n;

    //An array to mark processed array
    bool *processed = (bool *)malloc( numEdges * sizeof(bool) );
    assert( processed != NULL );

    long currTail = 0;
    long nextTail = 0;

    eid_t *curr = (eid_t *)malloc( numEdges * sizeof(eid_t) );
    assert( curr != NULL );

    eid_t *next = (eid_t *)malloc( numEdges * sizeof(eid_t) );
    assert( next != NULL );

    eid_t *startEdge = (eid_t *)malloc(n * sizeof(eid_t) );
    assert( startEdge != NULL );


    eid_t *X = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( X != NULL );

    for(vid_t i = 0; i < g->n; i++) {
        X[i] = 0;
    }

    //Initialize the arrays
    for( eid_t e = 0; e < numEdges; e++ ) {
        processed[ e ] = false;
    }

    //Find the startEdge for each vertex
    for( vid_t i = 0; i < n; i++ ) {
        eid_t j = g->num_edges[i];
        eid_t endIndex = g->num_edges[i+1];

        while( j < endIndex ) {
            if(g->adj[j] > i)
                break;
            j++;
        }
        startEdge[i] = j;
    }

#if TIME_RESULTS
    double triTime = 0;
    double scanTime = 0;
    double procTime = 0;
    double startTime = timer();
#endif

    for( vid_t u = 0; u < n; u++ ) {

        for(eid_t j = startEdge[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            X[w] = j +1;
        }

        for(eid_t j = g->num_edges[u]; j < startEdge[u]; j++) {
            vid_t v = g->adj[j];

            for(eid_t k = g->num_edges[v+1]-1; k >= startEdge[v]; k --) {
                vid_t w = g->adj[k];
                // check if: w > u
                if( w <= u ) {
                    break;
                }

                if(  X[w] ) {  //This is a triangle
                    //edge id's are: <u,w> : g->eid[ X[w] -1] 
                    //<u,w> : g->eid[ X[w] -1] 
                    //<v,u> : g->eid[ j ]  
                    //<v,w> : g->eid[ k ]   
                    eid_t e1 = g->eid[ X[w] -1 ] , e2 = g->eid[j], e3 = g->eid[k];
                    EdgeSupport[e1] += 1;
                    EdgeSupport[e2] += 1;
                    EdgeSupport[e3] += 1;
                }
            }
        }

        for(eid_t j = startEdge[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            X[w] = 0;
        }
    }

#if TIME_RESULTS
    triTime = timer() - startTime ;
    startTime = timer();
#endif

    //Support computation is done
    //Computing truss now

    int level = 0;
    long todo = numEdges;

    while( todo > 0 ) {

#if TIME_RESULTS
        startTime = timer();
#endif

        PKT_scan_serial(numEdges, EdgeSupport, level, curr, &currTail);

#if TIME_RESULTS
        scanTime += timer() - startTime;
        startTime = timer();
#endif

        while( currTail > 0 ) {
            todo = todo - currTail;

            PKT_processSubLevel_serial_marking(g, curr, currTail, EdgeSupport, level, next, &nextTail, X, processed, edgeIdToEdge);

            eid_t *tempCurr = curr;
            curr = next;
            next = tempCurr;

            currTail = nextTail;
            nextTail = 0;
        }

#if TIME_RESULTS
        procTime += timer() - startTime;
#endif

        level = level + 1;
    }

#if TIME_RESULTS
    printf("Tri time: %9.3lf \nScan Time: %9.3lf \nProc Time: %9.3lf \n", triTime, scanTime, procTime);
    printf("PKT-serial-Time-marking: %9.3lf\n", triTime + scanTime + procTime);
#endif

    free( X );

    //Free memory
    free( next );
    free( curr );
    free( processed );
    free( startEdge );
}



/**   Computes the support of each edge in parallel 
 *    Computes k-truss in parallel   ****/
void PKT_marking(graph_t *g, int *EdgeSupport, Edge *edgeIdToEdge) {

    long numEdges = g->m / 2;
    long n = g->n;

    bool *processed = (bool *)malloc( numEdges * sizeof(bool) );
    assert( processed != NULL );

    long currTail = 0;   
    long nextTail = 0;   

    eid_t *curr = (eid_t *)malloc( numEdges * sizeof(eid_t) ); 
    assert( curr != NULL ); 

    bool *InCurr = (bool *)malloc( numEdges * sizeof(bool) ); 
    assert( InCurr != NULL ); 

    eid_t *next = (eid_t *)malloc( numEdges * sizeof(eid_t) ); 
    assert( next != NULL );

    bool *InNext = (bool *)malloc( numEdges * sizeof(bool) ); 
    assert( InNext != NULL ); 

    eid_t *startEdge = (eid_t *)malloc(n * sizeof(eid_t) );
    assert( startEdge != NULL );

 
    //parallel region
#pragma omp parallel 
{ 
    int tid =omp_get_thread_num();

    eid_t *X = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( X != NULL );

    for(vid_t i = 0; i < g->n; i++) {
	X[i] = 0;
    }

#pragma omp for schedule(static) 
    for( eid_t e = 0; e < numEdges; e++ ) {
	//Initialize processed array with false
	processed[ e ] = false;
	
	InCurr[e] = false;
	InNext[e] = false;
    }



#pragma omp for schedule(static) 
    for( vid_t i = 0; i < n; i++ ) {
	eid_t j = g->num_edges[i];
	eid_t endIndex = g->num_edges[i+1];

	while( j < endIndex ) {
	    if(g->adj[j] > i)
		break;
	    j++;
	}
	startEdge[i] = j;
    } 

#if TIME_RESULTS
    double triTime = 0;
    double scanTime = 0;
    double procTime = 0;
    double start = timer();
#endif


#pragma omp for schedule(dynamic,10) 
    for( vid_t u = 0; u < n; u++ ) {

        for(eid_t j = startEdge[u]; j < g->num_edges[u+1]; j++) {
	    vid_t w = g->adj[j];
	    X[w] = j +1;
	}

        for(eid_t j = g->num_edges[u]; j < startEdge[u]; j++) {
	    vid_t v = g->adj[j];

	    for(eid_t k = g->num_edges[v+1]-1; k >= startEdge[v]; k --) {
                vid_t w = g->adj[k];
                // check if: w > u
                if( w <= u ) {
		    break;
		}

                if(  X[w] ) {  //This is a triangle
                    //edge id's are: <u,w> : g->eid[ X[w] -1] 
                    //<u,w> : g->eid[ X[w] -1] 
                    //<v,u> : g->eid[ j ]  
                    //<v,w> : g->eid[ k ]		
                    eid_t e1 = g->eid[ X[w] -1 ] , e2 = g->eid[j], e3 = g->eid[k];
                    __sync_fetch_and_add(&EdgeSupport[e1], 1);
                    __sync_fetch_and_add(&EdgeSupport[e2], 1);
                    __sync_fetch_and_add(&EdgeSupport[e3], 1);
                } 
	    }                   
	}
	
	for(eid_t j = startEdge[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            X[w] = 0;
        }
    }


#if TIME_RESULTS
    triTime = timer() - start ;
    start = timer();
#endif

    //Support computation is done
    //Computing truss now

    int level = 0;
    long todo = numEdges;
   
    while( todo > 0 ) {

#if TIME_RESULTS
	start = timer();
#endif

	PKT_scan(numEdges, EdgeSupport, level, curr, &currTail, InCurr);

#if TIME_RESULTS
	scanTime += timer() - start;
	start = timer();
#endif

	while( currTail > 0 ) {
	    todo = todo - currTail;
	
	    PKT_processSubLevel_marking(g, curr, InCurr, currTail, EdgeSupport, level, next, InNext, &nextTail, X, processed, edgeIdToEdge);

	    if( tid == 0 ) {
		eid_t *tempCurr = curr;
		curr = next;
		next = tempCurr;

		bool *tempInCurr = InCurr;
		InCurr = InNext;
		InNext = tempInCurr;


		currTail = nextTail;
		nextTail = 0;
	    }

#pragma omp barrier	
	}

#if TIME_RESULTS
	procTime += timer() - start;
#endif

	level = level + 1;
#pragma omp barrier	

    }


#if TIME_RESULTS
    if(tid == 0 ) {
        printf("Tri time: %9.3lf \nScan Time: %9.3lf \nProc Time: %9.3lf \n", triTime, scanTime, procTime);
	printf("PKT-marking-Time: %9.3lf\n\n\n", triTime + scanTime + procTime);
    }
#endif

    free( X );

}  //End of parallel region

    //Free memory
    free( next );
    free( InNext );
    free( curr );
    free( InCurr );
    free( processed );
    free( startEdge ); 
}


 
/**   Ros algorithms for truss decomposition 
 *    Computes the support of each edge in parallel 
 *    Computes k-truss in serial -- similar to WC algorithm 
 *
 *    Ryan A. Rossi, "Fast Triangle Core Decomposition for Mining Large Graphs", in Proc. 
 *    Pacific-Asia Conference on Advances in Knowledge Discovery and Data Mining (PAKDD), 2014
 *
 *    ****/
void Ros(graph_t *g, int *EdgeSupport, Edge * edgeIdToEdge) {

    long numEdges = g->m / 2;

    int *local_max_support = (int *)malloc(NUM_THREADS * sizeof(unsigned int));
    assert( local_max_support != NULL );

#if TIME_RESULTS
    double supTime = 0.0, procTime = 0.0;
    double startTime = timer();
#endif

    //parallel region
#pragma omp parallel 
{
    int tid = omp_get_thread_num();
    local_max_support[tid] = 0;

    eid_t *X = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( X != NULL );

    for(long i = 0; i < g->n; i++) {
        X[i] = numEdges+1;
    }

#if TIME_RESULTS
    if( tid == 0 ) {
	startTime = timer();
    }
#endif

    //Compute the support of each edge in parallel
#pragma omp for schedule(static) 
    for( long e = 0; e < numEdges; e++ ) {

        Edge edge = edgeIdToEdge[e];

        int support = 0;

        vid_t u = edge.u;
        vid_t v = edge.v;

        //find the number of elements in the intersection of N(u) and N(v)
        //This can be done in time d(u) + d(v) 
        for(eid_t j = g->num_edges[u]; j <  g->num_edges[u+1]; j++) {
            vid_t w = g->adj[ j ];

            if( w != v ) {
                X[w] = e;  //store edge id: e
            }
        }

        for(eid_t j = g->num_edges[v]; j <  g->num_edges[v+1]; j++) {
            vid_t w = g->adj[ j ];

            if( w != u ) {
		//Check if it is marked
                if( X[w] == e ) {
                    support ++;
                }
            }
        }

        EdgeSupport[ e ] = support;

        if( support > local_max_support[tid]  ) {
            local_max_support[tid] = support;
        }
    }

#if TIME_RESULTS
    if( tid  == 0 ) { 
	supTime = timer() - startTime;
    }
#endif

    //free X
    free( X );
  
}  //End of parallel region

    int maxSupport = 0;
    for(int tid = 0; tid < NUM_THREADS; tid ++) {
        if( maxSupport < local_max_support[tid] ) {
            maxSupport = local_max_support[tid];
        }
    }

    //Copmute k-truss using bin-sort
    //Sorted edges array according to increasing support
    eid_t *sortedEdges = (eid_t *)malloc( numEdges * sizeof(eid_t) );
    assert( sortedEdges != NULL );

    //Position of edge in sortedEdges
    eid_t *edgePos = (eid_t *)malloc( numEdges * sizeof(eid_t) );
    assert( edgePos != NULL );

    //number of bins is (maxSupport + 2)
    //the support is in: 0 ... maxSupport 
    //number of values: maxSupport +1
    unsigned int *bin = (unsigned int *)malloc((maxSupport +2)*sizeof(unsigned int));
    assert( bin != NULL);

    for(long i = 0; i < maxSupport +2; i++)
        bin[i] = 0;

#if TIME_RESULTS
    startTime = timer();
#endif

    //Find number of edges for each support in: 0...maxSupport
    for(long i = 0; i < numEdges; i++) {
        bin[ EdgeSupport[i] ] ++;
    }

    unsigned int start = 0;
    for(int support = 0; support  < maxSupport +1; support ++) {
        unsigned int num = bin[support];
        bin[support] = start;
        start = start + num;
    }

    //Do bin-sort/bucket-sort of the edges
    //sortedEdges -- contains the edges in increasing order of support
    //edgePos -- contains the position of an edge in sortedEdges array  
    for(long i = 0; i < numEdges; i++) {
        edgePos[ i ] = bin[  EdgeSupport[i] ];
        sortedEdges[ edgePos[i] ] = i;
        bin[ EdgeSupport[i] ] ++;
    }

    for(int d = maxSupport; d >= 1; d--)
        bin[d] = bin[d-1];
    bin[0] = 0;

    //STEP 3: Compute k-truss using support of each edge
    //an array to mark processed edges
    bool *proc = (bool *)malloc( numEdges * sizeof(bool) );
    assert( proc != NULL );

    eid_t *X = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( X != NULL );

    for(long i = 0; i < g->n; i++) {
        X[i] = 0;
    }

    for(long i = 0; i < numEdges; i++) {
        proc[i] = false;
    }

    //k-truss computations   
    //Edges are processed in increasing order of support
    for(long i = 0; i < numEdges; i++) {
        eid_t e = sortedEdges[i];

        Edge edge = edgeIdToEdge[e];
        vid_t u = edge.u;
        vid_t v = edge.v;

        for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            if(w != v) {
                X[w] = j+1; //position j in g->eid -- mark the neighbors of u
            }
        }

        for(eid_t j = g->num_edges[v]; j < g->num_edges[v+1]; j++) {
            vid_t w = g->adj[j];


            if( X[w] ) {  //if X[w] is marked
                eid_t e2 = g->eid[j]; //edge <v,w>
                eid_t e3 = g->eid[ X[w] -1]; //egde <u,w>

                if( ( proc[ e2 ] == false ) && ( proc[ e3 ] == false ) ) {  //if e, e2, e3 forms a triangle

                    for(int numAdj = 0; numAdj < 2; numAdj ++) {
                        eid_t edgeId = e2;
                        if(numAdj == 1)
                            edgeId = e3;

                        if( EdgeSupport[ edgeId] > EdgeSupport[e] ) {
                            int supportEid = EdgeSupport[ edgeId ];
                            unsigned int posEid = edgePos[ edgeId ];

                            unsigned int startPos = bin[ supportEid ]; //First position with support supportEid
                            unsigned int firstEdgeId = sortedEdges[ startPos ];

                            //Swap firstEdgeId and edgeId 
                            if( firstEdgeId != edgeId ) {
                                edgePos[ edgeId ] = startPos;
                                sortedEdges[ posEid ] = firstEdgeId;
                                edgePos[ firstEdgeId ] = posEid;
                                sortedEdges[ startPos ] = edgeId;
                            }

                            //Increase the starting index of bin[ supportEid ]
                            bin[ supportEid ] ++;

                            //Decrease support of edgeId -- so edgeId is in previous bin now     
                            EdgeSupport[edgeId] = EdgeSupport[edgeId] - 1;
                        }
                    }

                }

            }
        }

        proc[e] = true;
        for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            X[w] = 0; //reset to 0
        }

    }  //end of for loop      

#if TIME_RESULTS
    procTime = timer() - startTime;
    printf("support time: %9.3lf \nproc time: %9.3lf\n", supTime, procTime); 
    printf("Ros Time: %9.3lf\n", supTime + procTime); 
#endif

    /*****     Free Memory    ******/
    free( local_max_support );
    free( proc );
    free( X );
    free( sortedEdges );
    free( edgePos );
    free( bin );

}

/**   Serial Ros Algorithm ****
 *
 *    Ryan A. Rossi, "Fast Triangle Core Decomposition for Mining Large Graphs", in Proc. 
 *    Pacific-Asia Conference on Advances in Knowledge Discovery and Data Mining (PAKDD), 2014
 *
 *    */
void Ros_serial(graph_t *g, int *EdgeSupport, Edge * edgeIdToEdge) {

    long numEdges = g->m / 2;
    int maxSupport = 0;

    eid_t *X = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( X != NULL );

    for(long i = 0; i < g->n; i++) {
        X[i] = (numEdges + 1);
    }
    
#if TIME_RESULTS
    double supTime = 0, procTime = 0;
    double startTime = timer();
#endif

    //Compute the support of each edge 
    for( long e = 0; e < numEdges; e++ ) {

        Edge edge = edgeIdToEdge[e];

        int support = 0;

        vid_t u = edge.u;
        vid_t v = edge.v;

        //find the number of elements in the intersection of N(u) and N(v)
        //This can be done in time d(u) + d(v) 
        for(eid_t j = g->num_edges[u]; j <  g->num_edges[u+1]; j++) {
            vid_t w = g->adj[ j ];

            if( w != v ) {
                X[w] = e;  //store edge id: e
            }
        }

        for(eid_t j = g->num_edges[v]; j <  g->num_edges[v+1]; j++) {
            vid_t w = g->adj[ j ];

            if( w != u ) {
		//Check if it is marked
                if( X[w] == e ) {
                    support ++;
                }
            }
        }

        EdgeSupport[ e ] = support;

        if( maxSupport < support ) {
            maxSupport = support;
        }
    }

#if TIME_RESULTS
    supTime = timer() - startTime;
#endif

    //Copmute k-truss using bin-sort
    //Sorted edges array according to increasing support
    eid_t *sortedEdges = (eid_t *)malloc( numEdges * sizeof(eid_t) );
    assert( sortedEdges != NULL );

    //Position of edge in sortedEdges
    eid_t *edgePos = (eid_t *)malloc( numEdges * sizeof(eid_t) );
    assert( edgePos != NULL );

    //number of bins is (maxSupport + 2)
    //the support is in: 0 ... maxSupport 
    //number of values: maxSupport +1
    unsigned int *bin = (unsigned int *)malloc((maxSupport +2)*sizeof(unsigned int));
    assert( bin != NULL);

#if TIME_RESULTS
    startTime = timer();
#endif

    for(long i = 0; i < maxSupport +2; i++)
        bin[i] = 0;

    //Find number of edges for each support in: 0...maxSupport
    for(long i = 0; i < numEdges; i++) {
        bin[ EdgeSupport[i] ] ++;
    }

    unsigned int start = 0;
    for(int support = 0; support  < maxSupport +1; support ++) {
        unsigned int num = bin[support];
        bin[support] = start;
        start = start + num;
    }

    //Do bin-sort/bucket-sort of the edges
    //sortedEdges -- contains the edges in increasing order of support
    //edgePos -- contains the position of an edge in sortedEdges array  
    for(long i = 0; i < numEdges; i++) {
        edgePos[ i ] = bin[  EdgeSupport[i] ];
        sortedEdges[ edgePos[i] ] = i;
        bin[ EdgeSupport[i] ] ++;
    }

    for(int d = maxSupport; d >= 1; d--)
        bin[d] = bin[d-1];
    bin[0] = 0;

  
    //STEP 3: Compute k-truss using support of each edge 
    //an array to mark processed edges
    bool *proc = (bool *)malloc( numEdges * sizeof(bool) );
    assert( proc != NULL );

    for(long i = 0; i < g->n; i++) {
        X[i] = 0;
    }

    for(long i = 0; i < numEdges; i++) {
        proc[i] = false;
    }

    //k-truss computations
    //Edges are processed in increasing order of support
    for(long i = 0; i < numEdges; i++) {
        eid_t e = sortedEdges[i];

        Edge edge = edgeIdToEdge[e];
        vid_t u = edge.u;
        vid_t v = edge.v;

        for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            if(w != v) {
                X[w] = j+1; //position j in g->eid -- mark the neighbors of u
            }
        }

        for(eid_t j = g->num_edges[v]; j < g->num_edges[v+1]; j++) {
            vid_t w = g->adj[j];


            if( X[w] ) {  //if X[w] is marked
                eid_t e2 = g->eid[j]; //edge <v,w>
                eid_t e3 = g->eid[ X[w] -1]; //egde <u,w>

                if( ( proc[ e2 ] == false ) && ( proc[ e3 ] == false ) ) {  //if e, e2, e3 forms a triangle

                    for(int numAdj = 0; numAdj < 2; numAdj ++) {
                        eid_t edgeId = e2;
                        if(numAdj == 1)
                            edgeId = e3;

                        if( EdgeSupport[ edgeId] > EdgeSupport[e] ) {
                            int supportEid = EdgeSupport[ edgeId ];
                            unsigned int posEid = edgePos[ edgeId ];

                            unsigned int startPos = bin[ supportEid ]; //First position with support supportEid
                            unsigned int firstEdgeId = sortedEdges[ startPos ];

                            //Swap firstEdgeId and edgeId 
                            if( firstEdgeId != edgeId ) {
                                edgePos[ edgeId ] = startPos;
                                sortedEdges[ posEid ] = firstEdgeId;
                                edgePos[ firstEdgeId ] = posEid;
                                sortedEdges[ startPos ] = edgeId;
                            }

                            //Increase the starting index of bin[ supportEid ]
                            bin[ supportEid ] ++;

                            //Decrease support of edgeId -- so edgeId is in previous bin now
                            EdgeSupport[edgeId] = EdgeSupport[edgeId] - 1;
                        }
                    }

                }

            }
        }

        proc[e] = true;
        for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
            vid_t w = g->adj[j];
            X[w] = 0; //reset to 0
        }

    }  //end of for loop

#if TIME_RESULTS
    procTime = timer() - startTime;
    printf("support time: %9.3lf \nproc time: %9.3lf\n", supTime, procTime); 
    printf("Ros-serial Time: %9.3lf\n", supTime + procTime); 
#endif

    /*****     Free Memory    ******/
    free( proc );
    free( X );
    free( sortedEdges );
    free( edgePos );
    free( bin );

}

/**   WC algorithm for truss decomposition 
 *
 * J.Wang and J. Cheng, "Truss decomposition in massive networks", Proc. VLDB Endow., vol 5, no 9, pp. 
 * 812-823, May 2012.
 *
 * */
void WC(graph_t *g, int *EdgeSupport, MapType & edgeToIdMap, Edge * edgeIdToEdge) {
    long m = g->m;

    unsigned int *deg = (unsigned int *)malloc(g->n * sizeof(unsigned int));
    assert( deg != NULL );

    for(long i = 0; i < g->n; i++) {
        deg[i] = (g->num_edges[i+1] -  g->num_edges[i]);
    }


    long numEdges = m / 2 ;


    //STEP 1: Computer Support for each edge 
    //Maximum support of an edge
    int maxSupport = 0;

    eid_t *X = (eid_t *)malloc( g->n * sizeof(eid_t) );
    assert( X != NULL );

    for(long i = 0; i < g->n; i++) {
        X[i] = numEdges +1;
    }

#if TIME_RESULTS
    double procTime = 0.0, supTime = 0.0;
    double startTime = timer();
#endif

    //Compute the support of each edge
    for( long e = 0; e < numEdges; e++ ) {

        Edge edge = edgeIdToEdge[e];

        int support = 0;

        vid_t u = edge.u;
        vid_t v = edge.v;

        //find the number of elements in the intersection of nb(u) and nb(v)
        //This can be done in time |nb(u)| + |nb(v)|
        for(eid_t j = g->num_edges[u]; j <  g->num_edges[u+1]; j++) {
            vid_t w = g->adj[ j ];

            if( w != v ) {
                X[w] = e;  //store edge id -- e
            }
        }

        for(eid_t j = g->num_edges[v]; j <  g->num_edges[v+1]; j++) {
            vid_t w = g->adj[ j ];

            if( w != u ) {

                if( X[w] == e ) {
                    support ++;
                }
            }
        }

        EdgeSupport[ e ] = support;

        if( maxSupport < support ) {
            maxSupport = support;
        }
    }

#if TIME_RESULTS
    supTime = timer() - startTime;
#endif

    //free X
    free( X );


    //STEP 2: sort the edges in ascending order of their support -- using bin-sort
    eid_t *sortedEdges = (eid_t *)malloc( numEdges * sizeof(eid_t));
    assert( sortedEdges != NULL );

    eid_t *edgePos = (eid_t *)malloc( numEdges * sizeof(eid_t));
    assert( edgePos != NULL );

    //number of bins is (maxSupport + 2)
    //the support is in: 0 ... maxSupport 
    unsigned int *bin = (unsigned int *)calloc(maxSupport +2, sizeof(unsigned int));
    assert( bin != NULL);

    for(long i = 0; i < maxSupport +2; i++)
        bin[i] = 0;

#if TIME_RESULTS
    startTime = timer();
#endif

    //Find number of edges with each support in 0...maxSupport
    for(long i = 0; i < numEdges; i++) {
        bin[ EdgeSupport[i] ] ++;
    }

    unsigned int start = 0;
    for(int support = 0; support  < maxSupport +1; support ++) {
        unsigned int num = bin[support];
        bin[support] = start;
        start = start + num;
    }

    //Do bin-sort/bucket-sort of the edges
    //sortedEdges -- contains the edges in increasing order of support
    //edgePos -- contains the position of an edge in sortedEdges array  
    for(long i = 0; i < numEdges; i++) {
        edgePos[ i ] = bin[  EdgeSupport[i] ];
        sortedEdges[ edgePos[i] ] = i;
        bin[ EdgeSupport[i] ] ++;
    }

    for(int d = maxSupport; d >= 1; d--)
        bin[d] = bin[d-1];
    bin[0] = 0;

    //STEP 3: Compute k-truss using support of each edge
    int k = 2;

    //k-truss computations
    long i = 0;
    long numEdgesDeleted = 0;

    while( numEdgesDeleted < numEdges ) {

	//If edge sortedEdges[i] has support <= k-2         
        while( i < numEdges && EdgeSupport[  sortedEdges[i]  ] <= k-2 ) {
            eid_t e = sortedEdges[i];

            Edge edge = edgeIdToEdge[e];
            vid_t u = edge.u;
            vid_t v = edge.v;

            //Make sure: deg(u) < deg(v)
            if( deg[v] < deg[u] ) {
                //swap u and v
                vid_t temp = v;
                v = u;
                u = temp;
            }

            //Now, deg(u) < deg(v)
            for(eid_t j = g->num_edges[u]; j < g->num_edges[u+1]; j++) {
                vid_t w = g->adj[j];

                //Check if <u,w> is an edge or not -- it could be deleted
                Edge uw(u,w);
                if( w < u ) {
                    uw.u = w;
                    uw.v = u;
                }

		//If <u,w> is an edge
		MapType::iterator itUW = edgeToIdMap.find( uw );
                if( itUW != edgeToIdMap.end() ) {

                    //Check if <w,v> is an edge
                    Edge e(v,w);
                    if( w < v) {
                        e.u = w;
                        e.v = v;
                    }

		    //If <v,w> is an edge
		    MapType::iterator it = edgeToIdMap.find( e );
                    if( it != edgeToIdMap.end() ) {

                        //find edgeId of edge e(v,w)
                        eid_t edgeId =  it->second;

                        //If edge support of  <v,w> > k-2 then deccrement it
                        if( EdgeSupport[ edgeId] > k -2 ) {

                            //swap edge <v,w> with the first edge in the bin[ EdgeSupport[edgeId] ]
                            int supportWV = EdgeSupport[ edgeId ];
                            unsigned int posWV = edgePos[ edgeId ];

                            unsigned int startPos = bin[ supportWV ]; //First position with support supportWV
                            unsigned int firstEdgeId = sortedEdges[ startPos ];

                            //Swap firstEdgeId and edgeId                           
                            if( firstEdgeId != edgeId ) {
                                edgePos[ edgeId ] = startPos;
                                sortedEdges[ posWV ] = firstEdgeId;
                                edgePos[ firstEdgeId ] = posWV;
                                sortedEdges[ startPos ] = edgeId;
                            }

                            //Increase the starting index of bin[ supportWV ]                           
                            bin[ supportWV ] ++;

                            //Decrease support of edgeId -- so edgeId is in previous bin now
                            EdgeSupport[edgeId] = EdgeSupport[edgeId] - 1;
                        }

                        //find edgeId of edge <u,w>                             
                        edgeId =  itUW->second;

			//If edge support of  <u,w> > k-2 then deccrement it
                        if( EdgeSupport[ edgeId ] > k- 2) {

                            //swap edge <u,w> with the first edge in the bin[ EdgeSupport[edgeId] ]   
                            int supportUW = EdgeSupport[ edgeId ];
                            unsigned int posUW = edgePos[ edgeId ];
                            unsigned int startPosUW = bin[ supportUW ]; //First position with support supportUW
                            unsigned int firstEdge = sortedEdges[ startPosUW ];

                            //swap firstEdge and edgeId                         
                            if( edgeId != firstEdge ) {
                                edgePos[ edgeId ] = startPosUW;
                                sortedEdges[ posUW ] = firstEdge;
                                edgePos[ firstEdge ] = posUW;
                                sortedEdges[ startPosUW ] = edgeId;
                            }

                            //Increase the starting index of bin[ supportUW ]
                            bin[ supportUW ] ++;

                            //Decrease support of edgeId -- so edgeId is in previous bin now                   
                            EdgeSupport[edgeId] = EdgeSupport[edgeId] - 1;
                        }
                    }  //<v,w> is an edge
                } //<u,w> is an edge               
            } // w is in N(u)

            i++;
	    numEdgesDeleted++;

            //Delete edge 'e' from the graph
            edgeToIdMap.erase( edge ); 
        }
        k++;
    }

#if TIME_RESULTS
    procTime = timer() - startTime;
    printf("Support Time: %9.3lf \nProc Time: %9.3lf\n", supTime, procTime);
    printf("WC Time: %9.3lf\n", supTime + procTime);
#endif

}

void display_stats(int *EdgeSupport, long numEdges) {
    int minSup = INT_MAX;    
    int maxSup = 0;

    for(long i = 0; i < numEdges; i++) {
	if( minSup > EdgeSupport[i] ) {
	    minSup = EdgeSupport[i];
	}

	if( maxSup < EdgeSupport[i] ) {
	    maxSup = EdgeSupport[i];
	}
    }

    long numEdgesWithMinSup = 0, numEdgesWithMaxSup = 0;

    for(long i = 0; i < numEdges; i++) {
        if( EdgeSupport[i] == minSup ) {
            numEdgesWithMinSup ++;
        }

        if( EdgeSupport[i] == maxSup ) {
            numEdgesWithMaxSup ++;
        }
    }

    printf("\nMin-truss: %d\n#Edges in Min-truss: %ld\n\n", minSup+2, numEdgesWithMinSup);
    printf("Max-truss: %d\n#Edges in Max-truss: %ld\n\n", maxSup+2, numEdgesWithMaxSup);


}


int main(int argc, char *argv[]) {

    if( argc < 2 ) {
	fprintf(stderr, "%s <Graph file>\n", argv[0]);
	exit(1);
    }

    read_env();

    graph_t g;

    //load the graph from file
    load_graph_from_file(argv[1], &g);

    /************   Compute k - truss *****************************************/
    //edge list array
    Edge * edgeIdToEdge = (Edge *)malloc( (g.m/2) * sizeof(Edge) );
    assert( edgeIdToEdge != NULL );

    //Populate the edge list array
    getEidAndEdgeList(&g, edgeIdToEdge);
    
    int *EdgeSupport = (int *)calloc( g.m /2, sizeof(int) );
    assert( EdgeSupport != NULL );

#pragma omp parallel for 
    for(long i = 0; i < g.m /2; i++) {
        EdgeSupport[i] = 0;
    }

    PKT_intersection(&g, EdgeSupport, edgeIdToEdge);

    display_stats(EdgeSupport, g.m /2); 

    //Free memory
    free_graph( &g );

    if( edgeIdToEdge != NULL )
	free( edgeIdToEdge );

    if( EdgeSupport != NULL )
	free( EdgeSupport );

    return 0;
}

