#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define EPSILON 1e-6

// Μετατροπή πυκνού σε CSR
void convert_to_csr(int *dense, int n, int nnz, int **values, int **col_inds, int **row_ptr) {
    *values = (int *)malloc(nnz * sizeof(int));
    *col_inds = (int *)malloc(nnz * sizeof(int));
    *row_ptr = (int *)malloc((n + 1) * sizeof(int));

    if (!*values || !*col_inds || !*row_ptr) {
        fprintf(stderr, "Error allocating CSR memory\n");
        exit(1);
    }

    int count = 0;
    (*row_ptr)[0] = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dense[i * n + j] != 0) {
                (*values)[count] = dense[i * n + j];
                (*col_inds)[count] = j;
                count++;
            }
        }
        (*row_ptr)[i + 1] = count;
    }
}

void spmv_csr_kernel(int rows, int *val, int *col, int *rpt, double *x, double *y) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        int row_start = rpt[i];
        int row_end = rpt[i+1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += val[j] * x[col[j]];
        }
        y[i] = sum;
    }
}

void dense_mv_kernel(int rows, int n, int *matrix, double *x, double *y) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += matrix[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n, iterations;
    float sparsity;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) printf("Usage: %s <N> <Sparsity 0.0-1.0> <Iterations>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    n = atoi(argv[1]);
    sparsity = atof(argv[2]);
    iterations = atoi(argv[3]);

    // Timers
    double t_csr_build = 0, t_comm_start = 0, t_comm_end = 0;
    double t_calc_csr_start = 0, t_calc_csr_end = 0;
    double t_total_csr_start = 0, t_total_csr_end = 0; 
    double t_dense_total_start = 0, t_dense_total_end = 0;

    // Data Structures
    int *dense_matrix = NULL;
    double *vector_x = (double *)malloc(n * sizeof(double)); 
    double *vector_y_local = NULL; 
    
    // Αποτελέσματα για επαλήθευση
    double *final_result_csr = NULL;
    double *final_result_dense = NULL;

    // CSR structures Global 
    int *csr_val = NULL, *csr_col = NULL, *csr_row_ptr = NULL;
    int nnz_total = 0;

    // CSR structures Local
    int *loc_val = NULL, *loc_col = NULL, *loc_row_ptr = NULL;

    // Setup & Initialization 
    int remainder = n % size;
    int local_rows = n / size + (rank < remainder ? 1 : 0);
    
    // Arrays για Scatterv/Gatherv 
    int *scounts_rows = malloc(size * sizeof(int));
    int *displs_rows = malloc(size * sizeof(int));
    
    // Arrays μόνο για το rank 0
    int *scounts_nnz = NULL;
    int *displs_nnz = NULL;

    if (rank == 0) {
        scounts_nnz = malloc(size * sizeof(int));
        displs_nnz = malloc(size * sizeof(int));
        
        srand(time(NULL));
        dense_matrix = (int *)malloc(n * n * sizeof(int));
        
        int r_sum = 0;
        for(int i=0; i<size; i++) {
            scounts_rows[i] = n / size + (i < remainder ? 1 : 0);
            displs_rows[i] = r_sum;
            r_sum += scounts_rows[i];
        }

        // Δημιουργία Πυκνού & Αρχικού Διανύσματος
        for (int i = 0; i < n; i++) vector_x[i] = 1.0;
        
        for (int i = 0; i < n * n; i++) {
            if ((float)rand() / RAND_MAX > sparsity) {
                dense_matrix[i] = (rand() % 10) + 1;
                nnz_total++;
            } else {
                dense_matrix[i] = 0;
            }
        }
        
        final_result_csr = (double *)malloc(n * sizeof(double));
        final_result_dense = (double *)malloc(n * sizeof(double));
    } else {
        int r_sum = 0;
        for(int i=0; i<size; i++) {
            scounts_rows[i] = n / size + (i < remainder ? 1 : 0);
            displs_rows[i] = r_sum;
            r_sum += scounts_rows[i];
        }
    }

    vector_y_local = (double *)malloc(local_rows * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_total_csr_start = MPI_Wtime();

    if (rank == 0) {
        double t1 = MPI_Wtime();
        convert_to_csr(dense_matrix, n, nnz_total, &csr_val, &csr_col, &csr_row_ptr);
        t_csr_build = MPI_Wtime() - t1;

        // Υπολογισμός Send Counts για τα NNZ δεδομένα
        int current_row = 0;
        int nnz_sum = 0;
        for (int i = 0; i < size; i++) {
            int rows_for_proc = scounts_rows[i];
            int start_idx = csr_row_ptr[current_row];
            int end_idx = csr_row_ptr[current_row + rows_for_proc];
            
            scounts_nnz[i] = end_idx - start_idx;
            displs_nnz[i] = nnz_sum;
            nnz_sum += scounts_nnz[i];
            current_row += rows_for_proc;
        }
    }

    // Data Distribution Time
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_comm_start = MPI_Wtime();

    MPI_Bcast(vector_x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int my_nnz;
    MPI_Scatter(scounts_nnz, 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_nnz > 0) {
        loc_val = (int *)malloc(my_nnz * sizeof(int));
        loc_col = (int *)malloc(my_nnz * sizeof(int));
    }
    loc_row_ptr = (int *)malloc((local_rows + 1) * sizeof(int));

    MPI_Scatterv(csr_val, scounts_nnz, displs_nnz, MPI_INT, loc_val, my_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(csr_col, scounts_nnz, displs_nnz, MPI_INT, loc_col, my_nnz, MPI_INT, 0, MPI_COMM_WORLD);


    if(rank == 0) {
        int current_r = 0;
        for(int i=0; i<=local_rows; i++) loc_row_ptr[i] = csr_row_ptr[i];
        
        for(int p=1; p<size; p++) {
            int r_count = scounts_rows[p];
            current_r += scounts_rows[p-1];
            MPI_Send(&csr_row_ptr[current_r], r_count + 1, MPI_INT, p, 99, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(loc_row_ptr, local_rows + 1, MPI_INT, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int start_offset = loc_row_ptr[0];
        for(int i=0; i<=local_rows; i++) loc_row_ptr[i] -= start_offset;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_comm_end = MPI_Wtime();

    // CSR Calculation Time
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_calc_csr_start = MPI_Wtime();

    for (int iter = 0; iter < iterations; iter++) {
        spmv_csr_kernel(local_rows, loc_val, loc_col, loc_row_ptr, vector_x, vector_y_local);
        
        MPI_Allgatherv(vector_y_local, local_rows, MPI_DOUBLE, 
                       vector_x, scounts_rows, displs_rows, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_calc_csr_end = MPI_Wtime();
        t_total_csr_end = MPI_Wtime();
        // Αποθήκευση αποτελέσματος CSR για έλεγχο
        for(int i=0; i<n; i++) final_result_csr[i] = vector_x[i];
    }

    for(int i=0; i<n; i++) vector_x[i] = 1.0;
    
    int *loc_dense = (int *)malloc(local_rows * n * sizeof(int));
    int *scounts_int = NULL; 
    int *displs_int = NULL;

    if(rank == 0){
        scounts_int = malloc(size * sizeof(int));
        displs_int = malloc(size * sizeof(int));
        int sum=0;
        for(int i=0; i<size; i++){
            scounts_int[i] = scounts_rows[i] * n;
            displs_int[i] = sum;
            sum += scounts_int[i];
        }
    }

    MPI_Scatterv(dense_matrix, scounts_int, displs_int, MPI_INT, loc_dense, local_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) t_dense_total_start = MPI_Wtime();

    for (int iter = 0; iter < iterations; iter++) {
        dense_mv_kernel(local_rows, n, loc_dense, vector_x, vector_y_local);
        MPI_Allgatherv(vector_y_local, local_rows, MPI_DOUBLE, 
                       vector_x, scounts_rows, displs_rows, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        t_dense_total_end = MPI_Wtime();
        // Αποθήκευση αποτελέσματος Dense για έλεγχο
        for(int i=0; i<n; i++) final_result_dense[i] = vector_x[i];
    }
    if (rank == 0) {
        int errors = 0;
        for(int i=0; i<n; i++){
            if(fabs(final_result_csr[i] - final_result_dense[i]) > EPSILON){
                errors++;
                if(errors < 5) printf("Mismatch at index %d: CSR=%f, Dense=%f\n", i, final_result_csr[i], final_result_dense[i]);
            }
        }
        if(errors > 0) printf("WARNING: Found %d mismatches!\n", errors);
        else printf("VERIFICATION SUCCESS: CSR and Dense results match.\n");

        printf("RESULTS: N=%d, Sparsity=%.2f, Iter=%d, Procs=%d\n", n, sparsity, iterations, size);
        printf("Time_CSR_Build: %f\n", t_csr_build);
        printf("Time_CSR_Comm: %f\n", t_comm_end - t_comm_start);
        printf("Time_CSR_Calc: %f\n", t_calc_csr_end - t_calc_csr_start);
        printf("Time_Total_CSR: %f\n", t_total_csr_end - t_total_csr_start);
        printf("Time_Total_Dense: %f\n", t_dense_total_end - t_dense_total_start);
        
        free(dense_matrix); free(csr_val); free(csr_col); free(csr_row_ptr);
        free(scounts_int); free(displs_int); free(scounts_nnz); free(displs_nnz);
        free(final_result_csr); free(final_result_dense);
    }

    free(vector_x); free(vector_y_local);
    if(loc_val) free(loc_val); 
    if(loc_col) free(loc_col); 
    free(loc_row_ptr); free(loc_dense);
    free(scounts_rows); free(displs_rows);

    MPI_Finalize();
    return 0;
}
