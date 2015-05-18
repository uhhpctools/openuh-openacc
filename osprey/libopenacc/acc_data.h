/**
 * Author: Rengan Xu
 * University of Houston
 */

#ifndef __DATA_H__
#define __DATA_H__

#include "acc_common.h"
#include "acc_kernel.h"

typedef struct{
	void *host_addr;
	void *device_addr;
	size_t size;
	//TRANSFER_TYPE type; /*indicate in/out from device*/
} param_t;

extern vector param_list;
extern acc_hashmap* map;
extern cudaStream_t async_streams[12];
extern int MODULE_BASE;

extern void* __acc_malloc_handler(unsigned int size);

extern void __acc_free_handler(void* ptr);

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

extern void __accr_wait_some_or_all_stream(int async_expr);

extern void __accr_destroy_all_streams(void);

extern int acc_async_test(int);

extern int acc_async_test_all(void);

extern void acc_async_wait(int);

extern void acc_async_wait_all(void);

extern void* acc_malloc(unsigned int);

extern void acc_free(void*);
#endif
