
#ifndef __SSIM_H__
#define __SSIM_H__

#ifndef OUT
#define OUT
#endif

#ifdef __cplusplus
    #define DEFAULT(x) =x
    #define EXTERN_C extern "C"
#else
    #define DEFAULT(x) // =x
    #define EXTERN_C // extern "C"
#endif

#ifdef _WIN32
    #define DLL_API EXTERN_C __declspec(dllexport)
#else
    #define DLL_API EXTERN_C // __attribute__((dllexport))
#endif

typedef unsigned char Byte;
typedef signed int   int32;

DLL_API double MSE_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height);
DLL_API double MSE_Float(float* pDataX, float* pDataY, int step, int width, int height);

DLL_API float PSNR_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal DEFAULT(255));
DLL_API float PSNR_Float(float* pDataX, float* pDataY, int step, int width, int height, double maxVal DEFAULT(2.0));

DLL_API float SSIM_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int win_size DEFAULT(7), int maxVal DEFAULT(255));
DLL_API float SSIM_Float(float* pDataX, float* pDataY, int step, int width, int height, int win_size DEFAULT(7), double maxVal DEFAULT(2.0));


#endif

