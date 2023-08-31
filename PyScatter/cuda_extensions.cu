#include <cupy/complex.cuh>

#define PI 3.141592653589793f

__global__ void cuda_calculate_scattering(
    complex<float> * const diffraction,
    const int size,
    const float * const S,
    const float * const coord,
    const int coord_length,
    const float * const occupancy
)
{
    int S_index = blockIdx.x * blockDim.x + threadIdx.x;

    float dotp;
    complex<float> res;
    float res_real, res_imag;

    if (S_index < size) {
        for (int coord_index = 0; coord_index < coord_length; coord_index++) {
            dotp = coord[coord_index*3 + 0] * S[size*0 + S_index] +
                coord[coord_index*3 + 1] * S[size*1 + S_index] + 
                coord[coord_index*3 + 2] * S[size*2 + S_index];

            sincosf(2.0f * PI * dotp, &res_imag, &res_real);
            res_real *= occupancy[coord_index];
            res_imag *= occupancy[coord_index];
            res = complex<float>(res_real, res_imag);
            //res *= occupancy[coord_index];

            diffraction[S_index] += res;
        }
    }
}

// # for coord, occupancy in zip(element_coords, element_occupancy):
// #     coord_slice = (slice(None), ) + (None, )*len(S.shape[1:])
// #     dotp = (coord[coord_slice] * S).sum(axis=0)
// #     element_diff += (occupancy * numpy.exp(2j * numpy.pi * dotp))
