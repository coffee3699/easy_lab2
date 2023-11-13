#include "multiply.h"
#include <cstdio>
#include <thread>
#include <immintrin.h>
#include <pmmintrin.h>
#include <algorithm>

#define BLOCK_SIZE 32*32

typedef struct {
    int thread_id;
    double (*matrix1)[M];
    double (*matrix2)[P];
    double (*result_matrix)[P];
} ThreadData;

void block_matrix_multiply_sse3(double (*matrix1)[M], double (*matrix2)[P], double (*result_matrix)[P], int row_start, int row_end, int col_start, int col_end,int mid_start,int mid_end) {
    for (int i = row_start; i < row_end; i++) {
        double *temp_rm = result_matrix[i];
        for (int k = mid_start; k < mid_end; k++) {
            double temp_m1 = matrix1[i][k];
            double *temp_m2 = matrix2[k];

            int j = col_start;
            int no_need = (col_end - j) % 2;
            for(int next = 0; next < no_need; next++, j++)
                temp_rm[j] += temp_m1 * temp_m2[j];

            for (; j < col_end; j += 2) {
                __m128d m1 = _mm_set1_pd(temp_m1);
                __m128d m2 = _mm_loadu_pd(temp_m2 + j);
                __m128d rm = _mm_loadu_pd(temp_rm + j);
                __m128d a = _mm_mul_pd(m1, m2);
                __m128d b = _mm_add_pd(a, rm);
                _mm_storeu_pd(temp_rm + j, b);
            }
        }
    }
}

void block_matrix_multiply_avx512(double (*matrix1)[M], double (*matrix2)[P], double (*result_matrix)[P], int row_start, int row_end, int col_start, int col_end,int mid_start,int mid_end) {
    for (int i = row_start; i < row_end; i++) {
        double *temp_rm = result_matrix[i];
        for (int k = mid_start; k < mid_end; k++) {
            double temp_m1 = matrix1[i][k];
            double *temp_m2 = matrix2[k];

            int j = col_start;
            int no_need = (col_end - j) % 8;
            for(int next = 0; next < no_need; next++, j++)
                temp_rm[j] += temp_m1 * temp_m2[j];

            for (; j < col_end; j += 8) {
                __m512d m1 = _mm512_set1_pd(temp_m1);
                __m512d m2 = _mm512_loadu_pd(temp_m2 + j);
                __m512d rm = _mm512_loadu_pd(temp_rm + j);
                __m512d a = _mm512_mul_pd(m1, m2);
                __m512d b = _mm512_add_pd(a, rm);
                _mm512_storeu_pd(temp_rm + j, b);
            }
        }
    }
}

void* block_matrix_multiply(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int NUM_THREADS = std::thread::hardware_concurrency();
    int rows_per_thread = (N + NUM_THREADS - 1) / NUM_THREADS;
    int start_row = data->thread_id * rows_per_thread;
    int end_row = std::min(start_row + rows_per_thread, N);

    for (int i = start_row; i < end_row; i += BLOCK_SIZE) {
        int block_i_end = std::min(i + BLOCK_SIZE, end_row);
        for (int j = 0; j < P; j += BLOCK_SIZE) {
            int block_j_end = std::min(j + BLOCK_SIZE, P);
            for (int k = 0; k < M; k += BLOCK_SIZE) {
                int block_k_end = std::min(k + BLOCK_SIZE, M);
                block_matrix_multiply_avx512(data->matrix1, data->matrix2, data->result_matrix, i, block_i_end, j, block_j_end, k, block_k_end);
            }
        }
    }

    pthread_exit(NULL);
}

void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {

    // Initialize the result matrix to zero
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            result_matrix[i][j] = 0.0;
        }
    }

    int NUM_THREADS = std::thread::hardware_concurrency();
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].matrix1 = matrix1;
        thread_data[i].matrix2 = matrix2;
        thread_data[i].result_matrix = result_matrix;
        pthread_create(&threads[i], NULL, block_matrix_multiply, (void*)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}