#include <curand_kernel.h>

extern "C" {
__global__ void get_literals(const unsigned int *ta_state,
                             int number_of_clauses, int number_of_literals,
                             int number_of_state_bits, unsigned int *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned int number_of_ta_chunks = (number_of_literals - 1) / 32 + 1;

    for (int j = index; j < number_of_clauses; j += stride) {
        for (int k = index; k < number_of_literals; k += stride) {
            unsigned int ta_chunk = k / 32;
            unsigned int chunk_pos = k % 32;
            unsigned int pos = j * number_of_ta_chunks * number_of_state_bits +
                               ta_chunk * number_of_state_bits +
                               number_of_state_bits - 1;
            if ((ta_state[pos] & (1 << chunk_pos)) > 0) {
                // Increment the count of the literal in the result array.
                unsigned int result_pos = j * number_of_literals + k;
                result[result_pos] = 1;
            }
        }
    }
}
}
