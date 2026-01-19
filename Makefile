CC = mpicc
CFLAGS = -O3 -Wall
LDFLAGS = -lm
TARGET = mpi_spmv

all: $(TARGET)

$(TARGET): mpi_spmv.c
	$(CC) $(CFLAGS) -o $(TARGET) mpi_spmv.c $(LDFLAGS)

clean:
	rm -f $(TARGET)