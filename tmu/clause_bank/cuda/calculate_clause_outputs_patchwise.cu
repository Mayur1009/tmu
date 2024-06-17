#include <curand_kernel.h>

extern "C"
{
    __device__ inline void calculate_clause_output_patchwise(unsigned int *ta_state, int number_of_ta_chunks, int number_of_state_bits, unsigned int filter, unsigned int *output, unsigned int *Xi)
    {
        for (int patch = 0; patch < NUMBER_OF_PATCHES; ++patch) {
            output[patch] = 1;
            for (int k = 0; k < number_of_ta_chunks-1; k++) {
                unsigned int pos = k*number_of_state_bits + number_of_state_bits-1;
                output[patch] = output[patch] && (ta_state[pos] & Xi[patch*number_of_ta_chunks + k]) == ta_state[pos]; 

			if (!output[patch]) {
				break;
			}
		}

		unsigned int pos = (number_of_ta_chunks-1)*number_of_state_bits + number_of_state_bits-1;
		output[patch] = output[patch] &&
			(ta_state[pos] & Xi[patch*number_of_ta_chunks + number_of_ta_chunks - 1] & filter) ==
			(ta_state[pos] & filter);
        }
    }

    __global__ void calculate_clause_outputs_patchwise(unsigned int *ta_state, int number_of_clauses, int number_of_literals, int number_of_state_bits, unsigned int *clause_output, unsigned int *X, int e)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        unsigned int filter;
        if (((number_of_literals) % 32) != 0) {
            filter  = (~(0xffffffff << ((number_of_literals) % 32)));
        } else {
            filter = 0xffffffff;
        }
        unsigned int number_of_ta_chunks = (number_of_literals-1)/32 + 1;

        for (int j = index; j < number_of_clauses; j += stride) {
            unsigned int clause_pos = j*number_of_ta_chunks*number_of_state_bits;
            calculate_clause_output_patchwise(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, &clause_output[j*NUMBER_OF_PATCHES], &X[e*(number_of_ta_chunks*NUMBER_OF_PATCHES)]);
        }
    }
}
