# PKT
Shared-memory Graph Truss Decomposition

## Compiling the program:
    
    $ cd /PKT/
    $ make 
    By default, g++ compiler is used. To use Intel's icc compiler, please change the Makefile
    to include -qopenmp flag.
    
    
## Input Format:

    The input to the program is a text file. The file should contain an undirected graph and vertex index should 
    start from 0. The first line contains the number of vertices(N) and number of edges(M). Next M lines contain the M
    edges. An example graph is given in example.txt. 
    
## Execution:

    The program can be executed using the following command:
    $ OMP_NUM_THREADS=p ./PKT.exe graph.txt

    where, p is the number of threads used and graph.txt is a file containing a sparse graph. By default the number
    of threads is 1.
    
## Citing PKT:

    If you use PKT, please cite our paper:
     H. Kabir and K. Madduri, "Shared-memory Graph Truss Decomposition", arXiv.org e-Print archive, 
     https://arxiv.org/abs/1707.02000, July 2017.
    
## Support:

    Please email Humayun Kabir (hzk134@cse.psu.edu) and Kamesh Madduri (madduri@cse.psu.edu).
