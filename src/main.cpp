#pragma warning(disable:4996)
#include "ssim.h"
#include "stdio.h"
#include "string.h"

#ifdef _WIN32
#include "windows.h"
#define TIME_DECLARE() LARGE_INTEGER __freq, __t1, __t2; QueryPerformanceFrequency(&__freq)
#define TIME_BEGIN() QueryPerformanceCounter(&__t1)
#define TIME_END(x) QueryPerformanceCounter(&__t2); printf("%s: %.3f ms\n", x, 1000.0*(double)(__t2.QuadPart-__t1.QuadPart)/(double)__freq.QuadPart)
#else
#include<sys/time.h>
#include<unistd.h>
#define TIME_DECLARE() double __t1, __t2; timeval __tv
#define TIME_BEGIN() gettimeofday(&__tv, NULL); __t1=(double)(__tv.tv_sec*1000000.0 + __tv.tv_usec)
#define TIME_END(x) gettimeofday(&__tv, NULL); __t2=(double)(__tv.tv_sec*1000000.0 + __tv.tv_usec); printf("%s: %.3f ms\n", x, (__t2-__t1)/1000.0)
#endif

static int load_data(const char* pName, OUT Byte* pData)
{
    int n=0;
    FILE* fp=fopen(pName, "rb");
    if(fp)
    {
        fseek(fp, 0, SEEK_END);
        int flen=(int)ftell(fp);
        rewind(fp);
        n=(int)fread(pData, 1, flen, fp);
        fclose(fp);
    }
    return n;
}

int main()
{
    double psnr=0, ssim=0;
    int width=1920, height=1080, step=width*3, n=10;
    Byte *pDataX=new Byte[step*height], *pDataY=new Byte[step*height];
    load_data("../x_1920x1080.RGB", pDataX);
    load_data("../y_1920x1080.RGB", pDataY);
    TIME_DECLARE();
    TIME_BEGIN();
    for(int i=0; i<n; i++)
        psnr+=PSNR_Byte(pDataX, pDataY, step, width, height);
    TIME_END("PSNR");
    TIME_BEGIN();
    for(int i=0; i<n; i++)
        ssim+=SSIM_Byte(pDataX, pDataY, step, width, height);
    TIME_END("SSIM");
    printf("psnr=%f\n", psnr/n);
    printf("ssim=%f\n", ssim/n);
    delete[]pDataX;
    delete[]pDataY;
}
