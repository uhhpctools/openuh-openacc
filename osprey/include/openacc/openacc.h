
typedef enum
{
  acc_device_none     = 0, //< no device
  acc_device_default  = 1, //< default device type
  acc_device_host     = 2, //< host device
  acc_device_not_host = 3, //< not host device
  acc_device_cuda     = 4, //< CUDA device
  acc_device_opencl   = 5  //< OpenCL device
} acc_device_t;

typedef enum
{
	ACC_KDATA_UNKNOWN = 0,
	//MTYPE_I1 = 2        /* 8-bit integer */
	ACC_KDATA_UINT8,      
	//MTYPE_I2 = 3        /* 16-bit integer */
	ACC_KDATA_UINT16,
	//MTYPE_I4 = 4        /* 32-bit integer */
	ACC_KDATA_UINT32,
	//MTYPE_I8 = 5        /* 64-bit integer */
	ACC_KDATA_UINT64,
	//MTYPE_U1 = 6        /* 8-bit unsigned integer */
	ACC_KDATA_INT8,
	//MTYPE_U2 = 7        /* 16-bit unsigned integer */
	ACC_KDATA_INT16,
	//MTYPE_U4 = 8        /* 32-bit unsigned interger */
	ACC_KDATA_INT32,
	//MTYPE_U8 = 9        /* 64-bit unsigned integer */
	ACC_KDATA_INT64,
	//MTYPE_F4 = 10       /* 32-bit IEEE floating point */
	ACC_KDATA_FLOAT,
	//MTYPE_F8 = 11       /* 64-bit IEEE floating point */
	ACC_KDATA_DOUBLE
} ACC_KERNEL_DATA_TYPE;

extern void _w2c_mstore(void* src, int src_offset, void* dst, int dst_offset, int ilength);
extern void __accr_setup(void);

extern void __accr_cleanup(void);

extern void __accr_malloc_on_device(void* pHost, void** pDevice, unsigned int size);

extern void __accr_free_on_device(void* pDevice);

extern void __accr_memin_h2d(void* pHost, 
							 void* pDevice, 
							 unsigned int size, 
							 unsigned int offset,
							 int async_expr);

extern void __accr_memout_d2h(void* pDevice, 
							  void* pHost, 
							  unsigned int size, 
							  unsigned int offset,
							  int async_expr);

extern void __accr_init_param_list();

extern void __accr_push_kernel_param_pointer(void** pParam);

extern void __accr_push_kernel_param_scalar(void* pValue);

extern void __accr_push_kernel_param_int(int* iValue);

extern void __accr_push_kernel_param_float(float* ftValue);

extern void __accr_push_kernel_param_double(double* dbValue);

extern void __accr_clean_param_list();

extern int __accr_get_device_addr(void* pHostAddr, 
								   void** pDeviceAddr, 
								   unsigned int istart, 
								   unsigned int isize);

extern int __accr_present_create(void* pBuffer, 
								 unsigned int start, 
								 int length, 
								 unsigned int size);

extern int __accr_device_addr_present(void* pDevice);

extern void __accr_reduction_buff_malloc(void** pDevice, int type);

extern void __accr_update_device_variable(void* pHost, 
										  unsigned int offset, 
										  unsigned int size,
										  int async_expr);

extern void __accr_update_host_variable(void* pHost, 
										unsigned int offset, 
										unsigned int size,
										int async_expr);

extern void __accr_update_device_variable_async(void* pHost, unsigned int offset, unsigned int size, int scalar_expr);

extern void __accr_update_host_variable_async(void* pHost, unsigned int offset, unsigned int size, int scalar_expr);

extern void __accr_wait_stream(int async_expr);

extern void __accr_wait_all_streams(void);

extern int acc_async_test(int);

extern int acc_async_test_all(void);

extern void acc_async_wait(int);

extern void acc_async_wait_all(void);

extern void __accr_set_gangs(int x, int y, int z);

extern void __accr_set_vectors(int x, int y, int z);

extern void __accr_set_gang_num_x(int x);

extern void __accr_set_gang_num_y(int y);

extern void __accr_set_gang_num_z(int z);

extern void __accr_set_vector_num_x(int x);

extern void __accr_set_vector_num_y(int y);

extern void __accr_set_vector_num_z(int z);

extern void __accr_set_default_gang_vector(void);

extern void __accr_set_shared_mem_size(unsigned int size);

extern void __accr_set_default_shared_mem_size(void);

extern void __accr_reset_default_gang_vector(void);

extern int __accr_get_num_workers();

extern int __accr_get_num_vectors();

extern int __accr_get_total_num_gangs(void);

extern int __accr_get_total_gangs_workers();

extern int __accr_get_total_num_vectors();

extern void __accr_launchkernel(char* szKernelName, char* szKernelLib, int async_expr);

//extern void __accr_final_reduction_algorithm(double* result, double *d_idata, int type);
extern void __accr_final_reduction_algorithm(void* result, void *d_idata, char* kernel_name, unsigned int size, unsigned int type_size);

extern void acc_init(acc_device_t);

extern void acc_shutdown(acc_device_t);

extern void* acc_malloc(unsigned int);

extern void acc_free(void*);

extern void __acc_stack_push();
extern void __acc_stack_pop();
extern void __acc_stack_pending_to_current_stack(void* pdevice);
extern void __acc_stack_clear_device_ptr_in_current_stack();
