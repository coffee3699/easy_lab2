#include "multiply.h"
#include <cstdio>
#include <thread>
#include <vector>

const int BLOCK_SIZE = 32;
const int NUM_THREADS = 16;

// Function to rearrange matrix for better cache utilization
void rearrange_matrix(double src[M][P], double dest[M][P]) {
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int j = 0; j < P; j += BLOCK_SIZE) {
            for (int row = i; row < i + BLOCK_SIZE && row < M; ++row) {
                for (int col = j; col < j + BLOCK_SIZE && col < P; ++col) {
                    dest[row][col] = src[row][col];
                }
            }
        }
    }
}

void blocked_multiply(int startRow, int endRow,
                      double temp_matrix1[N][M],
                      double temp_matrix2[M][P],
                      double result_matrix[N][P]) {
    for (int i = startRow; i < endRow; i += BLOCK_SIZE)
        for (int j = 0; j < P; j += BLOCK_SIZE)
            for (int k = 0; k < M; k += BLOCK_SIZE)
                for (int row = i; row < i + BLOCK_SIZE && row < N; ++row)
                    for (int mid = k; mid < k + BLOCK_SIZE && mid < M; ++mid)
                        for (int col = j; col < j + BLOCK_SIZE && col < P; ++col)
                            result_matrix[row][col] += temp_matrix1[row][mid] * temp_matrix2[mid][col];
}

void matrix_multiplication(double matrix1[N][M], double matrix2[M][P], double result_matrix[N][P]) {

    double temp_matrix1[N][M];
    double temp_matrix2[M][P];

    rearrange_matrix(matrix1, temp_matrix1);
    rearrange_matrix(matrix2, temp_matrix2);

    std::vector<std::thread> threads;

    int rowsPerThread = N / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == NUM_THREADS - 1) ? N : startRow + rowsPerThread;

        threads.push_back(std::thread(blocked_multiply, startRow, endRow, temp_matrix1, temp_matrix2, result_matrix));
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
