import cupy
import pathlib

def import_cuda_file(filename, kernels):
    cuda_dir = pathlib.Path(__file__).parent.absolute()
    with open(cuda_dir / filename, 'r') as f:
        code = f.read()
    module = cupy.RawModule(code=code, options=('--std=c++11',),
                            name_expressions=kernels)
    return_dict = {}
    for this_kernel in kernels:
        return_dict[this_kernel] = module.get_function(this_kernel)
    return return_dict

kernels = import_cuda_file('cuda_extensions.cu', ['cuda_calculate_scattering'])

def calculate_scattering(element_diff, S, element_coords, element_occupancy):
    nthreads = 256
    nblocks = (element_diff.size - 1) // nthreads + 1

    arguments = (element_diff, element_diff.size, S, element_coords, element_coords.shape[0], element_occupancy)
    kernels["cuda_calculate_scattering"]((nblocks, ), (nthreads, ), arguments)