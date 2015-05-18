/**
 * Author: Rengan Xu
 * University of Houston
 */

#include "acc_context.h"
#include "acc_kernel.h"


context_t *context;
acc_gpu_config_t *__acc_gpu_config;

void __accr_setup(void)
{

	/* it may have other initializations */
	__acc_gpu_config_create(&__acc_gpu_config);

	
	context = (context_t*)malloc(sizeof(context_t));
	context->device_type = acc_device_cuda;
//	cuDeviceGetName(context->name, 512, 0);
	cuDeviceGet(&(context->cu_device), 0);
	cuCtxCreate(&(context->cu_context), 0, context->cu_device);
}

void acc_init(acc_device_t device_type)
{
	/*To do: add the support for different accel types */
	__accr_setup();

	/*
	 * used for asynchronous streams, the stream array has
	 * 12 elements, the last is reserved for default stream
	 */
	MODULE_BASE = 11;

    /* set up the default shared memory size */
    __accr_set_default_shared_mem_size();
}

int __acc_gpu_config_create(acc_gpu_config_t** _config)
{
	acc_gpu_config_t *config;
	int device_id;
	
	/* so far we just use the first device */
	device_id = 0;
	
	cuInit(0);

	config = acc_host_alloc_zero(sizeof(*config));

	cuDeviceGetCount(&(config->num_devices));
	if(config->num_devices == 0)
	{
		ERROR(("No device available, abort"));
	}
	
	cuDeviceComputeCapability(&(config->major), &(config->minor), device_id);
	
	cuDeviceTotalMem(&(config->total_global_mem), device_id);

	cuDeviceGetAttribute(&(config->total_constant_mem), CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device_id);

	cuDeviceGetAttribute(&(config->shared_mem_size), CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device_id);

	cuDeviceGetAttribute(&(config->regs_per_block), CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device_id);
	
	cuDeviceGetAttribute(&(config->max_threads_per_block), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device_id);

	cuDeviceGetAttribute(&(config->max_block_dim[0]), CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device_id);

	cuDeviceGetAttribute(&(config->max_block_dim[1]), CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device_id);
	
	cuDeviceGetAttribute(&(config->max_block_dim[2]), CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device_id);
	
	cuDeviceGetAttribute(&(config->max_grid_dim[0]), CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device_id);
	
	cuDeviceGetAttribute(&(config->max_grid_dim[1]), CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device_id);
	
	cuDeviceGetAttribute(&(config->max_grid_dim[2]), CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device_id);
	
	INFO(("Number of devices: %d", config->num_devices));
	INFO(("Total global memory size: %lu", config->total_global_mem));
	INFO(("Total constant memory size: %d", config->total_constant_mem));
	INFO(("Shared memory size per block: %d", config->shared_mem_size));
	INFO(("Number of registers per block: %d", config->regs_per_block));
	INFO(("Max threads per block: %d", config->max_threads_per_block));
	INFO(("Max sizes of each dimension of a block: %d, %d, %d", 
											config->max_block_dim[0], 
											config->max_block_dim[1], 
											config->max_block_dim[2]));
	INFO(("Max sizes of each dimension of a grid: %d, %d, %d", 
											config->max_grid_dim[0], 
											config->max_grid_dim[1], 
											config->max_grid_dim[2]));
	*_config = config;
	return ACC_OK;
}

void __acc_gpu_config_destroy(acc_gpu_config_t* config)
{
	acc_host_free(config);
}

void __accr_cleanup(void)
{
	CUresult ret;
	ret = cuCtxDestroy(context->cu_context);
	CUDA_CHECK(ret);

	__acc_gpu_config_destroy(__acc_gpu_config);
}

void acc_shutdown(acc_device_t device_type)
{
	__accr_destroy_all_streams();
	/*To do: add the support for different accel types*/
	__accr_cleanup();
}
