import cupy
import pathlib

def import_cuda_file(filename: pathlib.Path, kernels: list[str]
                     ) -> dict[str, cupy.RawKernel]:
    cuda_dir = pathlib.Path(__file__).parent.absolute()
    with open(cuda_dir / filename, 'r') as f:
        code = f.read()
    module = cupy.RawModule(code=code, options=('--std=c++11',),
                            name_expressions=kernels)
    return_dict = {}
    for this_kernel in kernels:
        return_dict[this_kernel] = module.get_function(this_kernel)
    return return_dict
