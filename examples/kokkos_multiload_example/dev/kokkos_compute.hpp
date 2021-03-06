#include<Kokkos_Core.hpp>
#include<stdio.h>
#include<string>
#include<iostream>
#include<iomanip>
#include<impl/Kokkos_Timer.hpp>

#ifdef ENABLE_CUDA
	#include<cuda_runtime.h>
	#include<Kokkos_Cuda.hpp>
	//#include<helper_cuda.h>
#endif


typedef Kokkos::DefaultExecutionSpace DeviceSpace;

void init(int dev){
	Kokkos::InitArguments args;
	args.num_threads=0;
	args.num_numa=0;
	args.device_id=0;
	
	#ifdef ENABLE_CUDA
	args.device_id=dev;
	#endif

	Kokkos::initialize(args);
}

void finalize(){
	Kokkos::finalize();
}

#ifdef ENABLE_CUDA
/*The CUDA kernel */
__global__ void vector_add_cu(float *out, float *a, float *b, int n){
	for(int i = 0; i < n; i++){
		out[i] = a[i] + b[i];
	}
}

/* Implementation of the function to be wrapped by Cython */
void addition(float *out, float *a, float *b, int N){
    
    float *d_a, *d_b, *d_out;    

    cudaMalloc((void**)&d_a, sizeof(float)*N);
    cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc((void**)&d_out, sizeof(float)*N);

    cudaMemcpy(d_a, a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*N, cudaMemcpyHostToDevice);

    vector_add_cu<<<1, 1>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float)*N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
#else
void addition(float *out, float *a, float *b, int N){
    for(int i = 0; i < N; ++i){
        out[i] = a[i] + b[i];
    }    
}
#endif


double kokkos_function_copy(double* array, const int N, const int dev_id){

	using h_view = typename Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

	//Wrap raw pointer in Kokkos View for easy management. 	     
	h_view host_array(array, N);
	
	//Allocate memory on device (no op if only host)
	//auto device_array = Kokkos::create_mirror_view(host_array);

	//Copy to device (no op if only host)
	//Kokkos::deep_copy(device_array, host_array);

	//Setting range policy and doing explicit copies isn't necessary. The above should work, but this is a safety
	#ifdef ENABLE_CUDA
		cudaSetDevice(dev_id);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		Kokkos::Cuda cuda1(stream);
		auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(cuda1, 0, N);

		//Explicit copies because the default execution space isn't being set correctly
		using d_view = typename Kokkos::View<double*, Kokkos::CudaSpace>;
		d_view device_array("device_array", N);
		Kokkos::deep_copy(device_array, host_array);
	#else
		auto range_policy = Kokkos::RangePolicy<Kokkos::Serial>(0, N);
		using d_view = typename Kokkos::View<double*, Kokkos::HostSpace>;
		d_view device_array("device_array", N);
		Kokkos::deep_copy(device_array, host_array);
	#endif

	double sum = 0.0;
	{
		Kokkos::parallel_reduce("Reduction", N, KOKKOS_LAMBDA(const int i, double& lsum){
			lsum += device_array(i);
		}, sum);

		Kokkos::fence();
	}

	#ifdef ENABLE_CUDA
		cudaStreamDestroy(stream);
	#endif

	return sum;
};

//Generic function that maps 1D array to double
//Here we implement a reduction-sum
double kokkos_function(double* array, const int N, const int dev_id){
	
	std::cout<< "Running on Device" << dev_id <<std::endl;
	
	#ifdef ENABLE_CUDA
		//const cudaInternalDevices &dev_info = CudaInternalDevices::singleton();
		//auto cuda_space = Kokkos::Cuda();

	#endif

	//Create cuda stream for current device
	//All ifdefs here are unnecessary, but is a good safety:
	//The kernel will not launch if cudaSetDevice does not match the device set in kokkos's internal singleton
	#ifdef ENABLE_CUDA
		cudaSetDevice(dev_id);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		Kokkos::Cuda cuda1(stream);
		auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(cuda1, 0, N);
	#else
		auto range_policy = Kokkos::RangePolicy<Kokkos::Serial>(0, N);
	#endif

	double sum = 0.0;
	{
	//Kokkos::Timer timer;	
	
	//Turn array into Unmanaged Kokkos View in Default Exec Space
	Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Unmanaged>> array_view(array, N);

	//Launch Kokkos Kernel (Perform reduction)
	//Note that the range policy could be left as default and just specify N. 
	Kokkos::parallel_reduce(range_policy, KOKKOS_LAMBDA(const int i, double& lsum){
		//lsum += array[i]; //This also works (but is less robust for debugging)
		lsum += array_view(i);
	}, sum);

	Kokkos::fence();
	}
	//std::cout << "Finished on Device " << dev_id << std::endl;

	#ifdef ENABLE_CUDA
		cudaStreamDestroy(stream);
	#endif
	return sum;
};

