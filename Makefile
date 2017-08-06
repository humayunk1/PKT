CC=g++ 
#CC = icc 

CFLAGS = -fopenmp -std=c++0x -Wall -O3
#CFLAGS = -qopenmp -std=c++0x -Wall -O3


all: PKT

PKT:
	$(CC) $(CFLAGS) PKT.cpp -o ./PKT.exe

clean:
	rm -f ./*.exe ./*.o
