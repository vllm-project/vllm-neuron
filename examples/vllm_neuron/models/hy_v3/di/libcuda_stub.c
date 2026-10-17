/* Minimal libcuda.so.1 stub so NIXL's CUDA-linked LIBFABRIC plugin can load on a
 * host without an NVIDIA driver. Every entry point reports CUDA_ERROR_NO_DEVICE,
 * so the plugin falls back to its non-CUDA (Neuron / host) memory paths. */
typedef int CUresult;
#define CUDA_ERROR_NO_DEVICE 100
CUresult cuCtxSetCurrent(void *ctx) { (void)ctx; return CUDA_ERROR_NO_DEVICE; }
CUresult cuDeviceGetPCIBusId(char *bus, int len, int dev) { (void)bus; (void)len; (void)dev; return CUDA_ERROR_NO_DEVICE; }
CUresult cuPointerGetAttributes(unsigned n, void *attrs, void **data, unsigned long long ptr) {
  (void)n; (void)attrs; (void)data; (void)ptr; return CUDA_ERROR_NO_DEVICE; }
CUresult cuInit(unsigned flags) { (void)flags; return CUDA_ERROR_NO_DEVICE; }
CUresult cuDriverGetVersion(int *v) { if (v) *v = 0; return CUDA_ERROR_NO_DEVICE; }
CUresult cuDeviceGetCount(int *c) { if (c) *c = 0; return CUDA_ERROR_NO_DEVICE; }
CUresult cuGetProcAddress(const char *s, void **p, int v, unsigned long long f) {
  (void)s; (void)v; (void)f; if (p) *p = 0; return CUDA_ERROR_NO_DEVICE; }
CUresult cuGetProcAddress_v2(const char *s, void **p, int v, unsigned long long f, void *st) {
  (void)s; (void)v; (void)f; (void)st; if (p) *p = 0; return CUDA_ERROR_NO_DEVICE; }
