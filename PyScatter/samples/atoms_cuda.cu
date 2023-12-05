#include <cupy/complex.cuh>

#define PI 3.141592653589793f

__device__ void mul_bfactor(
    float &res_real,
    float &res_imag,
    const float * const S,
    const float bfactor
)
{
    float S_norm_square, bfactor_val;

    S_norm_square = S[0]*S[0] + S[1]*S[1] + S[2]*S[2];
    bfactor_val = expf(-bfactor * S_norm_square * 0.25f);
    
    res_real *= bfactor_val;
    res_imag *= bfactor_val;
}
    

__global__ void calculate_scattering(
    complex<float> * const diffraction,
    const int size,
    const float * const S,
    const float * const coord,
    const int coord_length,
    const float * const occupancy,
    const bool use_bfactor,
    const float * const bfactor
)
{
    int S_index = blockIdx.x * blockDim.x + threadIdx.x;
    float dotp;
    complex<float> res;
    float res_real, res_imag;

    if (S_index < size) {
        for (int coord_index = 0; coord_index < coord_length; coord_index++) {
            dotp = coord[coord_index*3 + 0] * S[S_index*3 + 0] +
                coord[coord_index*3 + 1] * S[S_index*3 + 1] + 
                coord[coord_index*3 + 2] * S[S_index*3 + 2];

            sincosf(2.0f * PI * dotp, &res_imag, &res_real);
            res_real *= occupancy[coord_index];
            res_imag *= occupancy[coord_index];

            if (use_bfactor) {
                mul_bfactor(res_real, res_imag, &(S[S_index*3]), bfactor[coord_index]);
            }

            res = complex<float>(res_real, res_imag);

            diffraction[S_index] += res;

        }
    }
}