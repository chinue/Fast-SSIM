#include "ssim.h"
#include "math.h"
#include "stdio.h"
#include "emmintrin.h"
//#include "immintrin.h" // AVX
#include "string.h"

//#include "Base.h"
#define MAX(a, b) ((a)<(b)?(b):(a))
#define MIN(a, b) ((a)<(b)?(a):(b))
#define ABS(a) ((a)<0?-(a):(a))
#define SQR(x) ((x)*(x))
#define AlignBytes(x) (((x)+15)/16*16)

static inline __m128 GetF32x4(__m128i u16)
{
    __m128i zero=_mm_setzero_si128();
    __m128i x8_0 = _mm_unpacklo_epi8(u16, zero);
    return _mm_cvtepi32_ps(_mm_unpacklo_epi16(x8_0, zero));
}

static inline __m128 LoadF32x4(Byte pData[8])
{
    return GetF32x4(_mm_loadl_epi64((__m128i*)pData));
}

template <typename Data, typename T>
static double MSE_Data_t(Data* pDataX, Data* pDataY, int widthBytes, int width, int height)
{
    double sum=0;
    int cn=widthBytes/width, width3=cn*width, width4=width3/4*4;
    for(int y=0; y<height; y++)
    {
        Data *pX=pDataX+y*widthBytes;
        Data *pY=pDataY+y*widthBytes;
        T s0=0, s1=0, s2=0, s3=0;
        for(int x=0; x<width4; x+=4)
        {
            T diff0=pY[x+0]-pX[x+0];
            T diff1=pY[x+1]-pX[x+1];
            T diff2=pY[x+2]-pX[x+2];
            T diff3=pY[x+3]-pX[x+3];
            //sum += diff*diff;
            s0 += diff0*diff0;
            s1 += diff1*diff1;
            s2 += diff2*diff2;
            s3 += diff3*diff3;
        }
        s0 += s1+s2+s3;
        for(int x=width4; x<width3; x++)
        {
            T diff0=pY[x]-pX[x];
            s0 += diff0*diff0;
        }
        sum += s0;
        //sum += s0+s1+s2+s3;
    }
    double mse=sum/(height*width3);
    return mse;
}
static double MSE_Data_Byte(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height)
{
    double sum=0;
    int cn=widthBytes/width, width3=cn*width, width4=width3/4*4;
    for(int y=0; y<height; y++)
    {
        Byte *pX=pDataX+y*widthBytes;
        Byte *pY=pDataY+y*widthBytes;
        int x=0;
        int width8=width3/8*8;
        __m128i zero=_mm_setzero_si128();
        __m128i s4 = _mm_setzero_si128();
        int s0=0;
        for(x=0; x<width8; x+=8)
        {
            __m128i x8 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(pX+x)), zero);
            __m128i y8 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(pY+x)), zero);
            __m128i d8 = _mm_sub_epi16(x8, y8);
            __m128i z8 = _mm_mullo_epi16(d8, d8);
            __m128i z8_0 = _mm_unpacklo_epi16(z8, zero);
            __m128i z8_1 = _mm_unpackhi_epi16(z8, zero);
            s4 = _mm_add_epi32(s4, _mm_add_epi32(z8_0, z8_1));
        }
        int* pS=(int*)&s4;
        s0=pS[0]+pS[1]+pS[2]+pS[3];
        for(; x<width3; x++)
        {
            int d=pY[x]-pX[x];
            s0 += d*d;
        }
        sum += s0;
    }
    double mse=sum/(height*width3);
    return mse;
}
double MSE_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height)
{
    return MSE_Data_Byte(pDataX, pDataY, step, width, height);
}
double MSE_Float(float* pDataX, float* pDataY, int step, int width, int height)
{
    return MSE_Data_t<float, float>(pDataX, pDataY, step, width, height);
}

template <typename Data, typename T>
static float PSNR_Data_t(Data* pDataX, Data* pDataY, int widthBytes, int width, int height, double maxVal)
{
    double mse = MSE_Data_t<Data, T>(pDataX, pDataY, widthBytes, width, height);
   //double maxVal=255;
   mse = MAX(mse, 1e-10);
   return (float)(10*log10(maxVal*maxVal/mse));
}
static float PSNR_Data_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal)
{
    double mse=MSE_Data_Byte(pDataX, pDataY, step, width, height);
    mse = MAX(mse, 1e-10);
    return (float)(10*log10(maxVal*maxVal/mse));
}
float PSNR_Float(float* pDataX, float* pDataY, int step, int width, int height, double maxVal)
{
    return PSNR_Data_t<float, float>(pDataX, pDataY, step, width, height, maxVal);
}
float PSNR_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal)
{
    return PSNR_Data_Byte(pDataX, pDataY, step, width, height, maxVal);
}

template<typename Data>
struct WinSum3F
{
    float *pSumX, *pSumY;
    float *pSumXX, *pSumXY, *pSumYY;
    int win_size;
    int widthBytes, width, height;
    Data *pDataX, *pDataY;
    int bufLen, cn;
    float maxVal;
    WinSum3F()
    {
        memset(this, 0, sizeof(*this));
    }
    WinSum3F(Data* pDataX, Data* pDataY, int widthBytes, int width, int height, int win_size, double maxVal)
    {
        this->Create(pDataX, pDataY, widthBytes, width, height, win_size, maxVal);
    }
    ~WinSum3F()
    {
        this->Release();
    }
    inline void Create(Data* pDataX, Data* pDataY, int widthBytes, int width, int height, int win_size, double maxVal)
    {
        this->win_size=win_size;
        this->width=width;
        this->height=height;
        this->cn=widthBytes/width;
        this->bufLen=AlignBytes((width+win_size)*cn+32);
        this->pSumX=new float[bufLen*5];
        this->pSumY=this->pSumX+bufLen;
        this->pSumXX=this->pSumX+bufLen*2;
        this->pSumXY=this->pSumX+bufLen*3;
        this->pSumYY=this->pSumX+bufLen*4;
        this->pDataX=pDataX;
        this->pDataY=pDataY;
        this->widthBytes=widthBytes;
        this->maxVal=(float)maxVal;
        for(int i=0; i<bufLen*5; i++)
            pSumX[i]=0;
    }
    inline void Release()
    {
        if(this->pSumX)
        {
            delete[] this->pSumX;
        }
        this->pSumX=0;
    }
    float DoFilter1()
    {
        float k1=0.01f, k2=0.03f, c1=SQR(k1*maxVal), c2=SQR(k2*maxVal);
        int x, y;
        int half_size=win_size/2;
        float invA=(float)(1.0/(win_size*win_size));
        float conv_norm=(float)(win_size*win_size)/(win_size*win_size-1);
        double sum=0;
        int count=0;
        InitFirstRow();
        for(y=half_size; y<height-half_size; y++)
        {
            float s=0;
            float sumX=0, sumY=0, sumXX=0, sumXY=0, sumYY=0;
            for(int i=0; i<win_size; i++)
            {
                sumX+=pSumX[i];  sumY+=pSumY[i];
                sumXX+=pSumXX[i]; sumXY+=pSumXY[i]; sumYY+=pSumYY[i];
            }
            for(x=half_size; x<width-half_size; x++)
            {
                float meanX=sumX*invA, meanY=sumY*invA;
                float meanXX=meanX*meanX, meanXY=meanX*meanY, meanYY=meanY*meanY;
                float sigmaXX=conv_norm*(sumXX*invA-meanXX);
                float sigmaXY=conv_norm*(sumXY*invA-meanXY);
                float sigmaYY=conv_norm*(sumYY*invA-meanYY);
                float u1=2*meanXY+c1, u2=meanXX+meanYY+c1;
                float v1=2*sigmaXY+c2, v2=sigmaXX+sigmaYY+c2;
                float z = (u1*v1)/(u2*v2);
                //sum += z;
                s += z;
                count++;

                // update row
                int i=(x+half_size+1), j=(x-half_size);
                sumX += pSumX[i]-pSumX[j];
                sumY += pSumY[i]-pSumY[j];
                sumXX += pSumXX[i]-pSumXX[j];
                sumXY += pSumXY[i]-pSumXY[j];
                sumYY += pSumYY[i]-pSumYY[j];
                
            }
            sum += s;
            UpdateNextRow(y+1);
        }
        return (float)(sum/count);
    }
    float DoFilter3()
    {
        float k1=0.01f, k2=0.03f, c1=SQR(k1*maxVal), c2=SQR(k2*maxVal);
        int x, y;
        int half_size=win_size/2;
        float invA=(float)(1.0/(win_size*win_size));
        float conv_norm=(float)(win_size*win_size)/(win_size*win_size-1);
        double sum=0;
        int count=0;
        InitFirstRow();
        for(y=half_size; y<height-half_size; y++)
        {
            __m128 s4 = _mm_set_ps1(0);
            __m128 invA4=_mm_set_ps1(invA), conv_norm4=_mm_set_ps1(conv_norm);
            __m128 sumX4 = _mm_loadu_ps(pSumX), sumY4 = _mm_loadu_ps(pSumY);
            __m128 sumXX4 = _mm_loadu_ps(pSumXX), sumXY4 = _mm_loadu_ps(pSumXY), sumYY4 = _mm_loadu_ps(pSumYY);
            for(int i=3; i<win_size*3; i+=3)
            {
                sumX4 = _mm_add_ps(_mm_loadu_ps(pSumX+i), sumX4);
                sumY4 = _mm_add_ps(_mm_loadu_ps(pSumY+i), sumY4);
                sumXX4 = _mm_add_ps(_mm_loadu_ps(pSumXX+i), sumXX4);
                sumXY4 = _mm_add_ps(_mm_loadu_ps(pSumXY+i), sumXY4);
                sumYY4 = _mm_add_ps(_mm_loadu_ps(pSumYY+i), sumYY4);
            }
            for(x=half_size; x<width-half_size; x++)
            {
                __m128 meanX4 = _mm_mul_ps(sumX4, invA4);
                __m128 meanY4 = _mm_mul_ps(sumY4, invA4);
                __m128 meanXX4 = _mm_mul_ps(meanX4, meanX4);
                __m128 meanXY4 = _mm_mul_ps(meanX4, meanY4);
                __m128 meanYY4 = _mm_mul_ps(meanY4, meanY4);
                __m128 sigmaXX4 = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(sumXX4, invA4), meanXX4), conv_norm4);
                __m128 sigmaXY4 = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(sumXY4, invA4), meanXY4), conv_norm4);
                __m128 sigmaYY4 = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(sumYY4, invA4), meanYY4), conv_norm4);

                __m128 c1_4 = _mm_set_ps1(c1), c2_4 = _mm_set_ps1(c2);
                __m128 u1_4 = _mm_add_ps(_mm_add_ps(meanXY4, meanXY4), c1_4);
                __m128 u2_4 = _mm_add_ps(_mm_add_ps(meanXX4, meanYY4), c1_4);
                __m128 v1_4 = _mm_add_ps(_mm_add_ps(sigmaXY4, sigmaXY4), c2_4);
                __m128 v2_4 = _mm_add_ps(_mm_add_ps(sigmaXX4, sigmaYY4), c2_4);
                __m128 z1_4 = _mm_mul_ps(u1_4, v1_4), z2_4 = _mm_mul_ps(u2_4, v2_4);
                __m128 z4 = _mm_div_ps(z1_4, z2_4);

                s4 = _mm_add_ps(s4, z4);
                count += 3;

                // update next column
                int i=3*(x+half_size+1), j=3*(x-half_size);
                sumX4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumX+i), _mm_loadu_ps(pSumX+j)), sumX4);
                sumY4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumY+i), _mm_loadu_ps(pSumY+j)), sumY4);
                sumXX4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumXX+i), _mm_loadu_ps(pSumXX+j)), sumXX4);
                sumXY4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumXY+i), _mm_loadu_ps(pSumXY+j)), sumXY4);
                sumYY4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumYY+i), _mm_loadu_ps(pSumYY+j)), sumYY4);
            }
            float *pS=(float*)&s4;
            sum += pS[0]+pS[1]+pS[2];
            if(sizeof(Data)==1)
                UpdateNextRow_Byte(y+1);
            else
                UpdateNextRow(y+1);
        }
        return (float)(sum/count);
    }
    float DoFilter()
    {
        if(cn==3)
            return this->DoFilter3();
        return this->DoFilter1();
    }
private:
    inline void InitFirstRow()
    {
        for(int y=0; y<win_size; y++)
        {
            Data *pX=pDataX+y*widthBytes;
            Data *pY=pDataY+y*widthBytes;
            for(int x=0; x<widthBytes; x++)
            {
                pSumX[x]+=pX[x];
                pSumY[x]+=pY[x];
                pSumXX[x]+=pX[x]*pX[x];
                pSumXY[x]+=pX[x]*pY[x];
                pSumYY[x]+=pY[x]*pY[x];
            }
        }
    }
    inline void UpdateNextRow(int y)
    {
        int half_win=win_size/2;
        if(y+half_win<height)
        {
            Data *pX0=pDataX+(y-1-half_win)*widthBytes, *pX1=pDataX+(y+half_win)*widthBytes;
            Data *pY0=pDataY+(y-1-half_win)*widthBytes, *pY1=pDataY+(y+half_win)*widthBytes;
            int width4=widthBytes/4*4, x=0;
            for(x=0; x<width4; x+=4)
            {
                __m128 x0_4 = _mm_set_ps(pX0[x+3], pX0[x+2], pX0[x+1], pX0[x+0]);
                __m128 x1_4 = _mm_set_ps(pX1[x+3], pX1[x+2], pX1[x+1], pX1[x+0]);
                __m128 y0_4 = _mm_set_ps(pY0[x+3], pY0[x+2], pY0[x+1], pY0[x+0]);
                __m128 y1_4 = _mm_set_ps(pY1[x+3], pY1[x+2], pY1[x+1], pY1[x+0]);
                _mm_storeu_ps(pSumX+x, _mm_add_ps(_mm_loadu_ps(pSumX+x), _mm_sub_ps(x1_4, x0_4)));
                _mm_storeu_ps(pSumY+x, _mm_add_ps(_mm_loadu_ps(pSumY+x), _mm_sub_ps(y1_4, y0_4)));
                _mm_storeu_ps(pSumXX+x, _mm_add_ps(_mm_loadu_ps(pSumXX+x), _mm_sub_ps(_mm_mul_ps(x1_4, x1_4), _mm_mul_ps(x0_4, x0_4))));
                _mm_storeu_ps(pSumXY+x, _mm_add_ps(_mm_loadu_ps(pSumXY+x), _mm_sub_ps(_mm_mul_ps(x1_4, y1_4), _mm_mul_ps(x0_4, y0_4))));
                _mm_storeu_ps(pSumYY+x, _mm_add_ps(_mm_loadu_ps(pSumYY+x), _mm_sub_ps(_mm_mul_ps(y1_4, y1_4), _mm_mul_ps(y0_4, y0_4))));
            }
            for(; x<widthBytes; x++)
            {
                float x1=pX1[x], x0=pX0[x], y0=pY0[x], y1=pY1[x];
                pSumX[x] += x1-x0;
                pSumY[x] += y1-y0;
                pSumXX[x]+= x1*x1-x0*x0;
                pSumXY[x]+= x1*y1-x0*y0;
                pSumYY[x]+= y1*y1-y0*y0;
            }
        }
    }
    inline void UpdateNextRow_Byte(int y)
    {
        int half_win=win_size/2;
        if(y+half_win<height)
        {
            Data *pX0=pDataX+(y-1-half_win)*widthBytes, *pX1=pDataX+(y+half_win)*widthBytes;
            Data *pY0=pDataY+(y-1-half_win)*widthBytes, *pY1=pDataY+(y+half_win)*widthBytes;
            int width4=(widthBytes-8)/4*4, x=0;
            for(x=0; x<width4; x+=4)
            {
                __m128 x0_4 = LoadF32x4((Byte*)(pX0+x));
                __m128 x1_4 = LoadF32x4((Byte*)(pX1+x));
                __m128 y0_4 = LoadF32x4((Byte*)(pY0+x));
                __m128 y1_4 = LoadF32x4((Byte*)(pY1+x));
                _mm_storeu_ps(pSumX+x, _mm_add_ps(_mm_loadu_ps(pSumX+x), _mm_sub_ps(x1_4, x0_4)));
                _mm_storeu_ps(pSumY+x, _mm_add_ps(_mm_loadu_ps(pSumY+x), _mm_sub_ps(y1_4, y0_4)));
                _mm_storeu_ps(pSumXX+x, _mm_add_ps(_mm_loadu_ps(pSumXX+x), _mm_sub_ps(_mm_mul_ps(x1_4, x1_4), _mm_mul_ps(x0_4, x0_4))));
                _mm_storeu_ps(pSumXY+x, _mm_add_ps(_mm_loadu_ps(pSumXY+x), _mm_sub_ps(_mm_mul_ps(x1_4, y1_4), _mm_mul_ps(x0_4, y0_4))));
                _mm_storeu_ps(pSumYY+x, _mm_add_ps(_mm_loadu_ps(pSumYY+x), _mm_sub_ps(_mm_mul_ps(y1_4, y1_4), _mm_mul_ps(y0_4, y0_4))));
            }
            for(; x<widthBytes; x++)
            {
                float x1=pX1[x], x0=pX0[x], y0=pY0[x], y1=pY1[x];
                pSumX[x] += x1-x0;
                pSumY[x] += y1-y0;
                pSumXX[x]+= x1*x1-x0*x0;
                pSumXY[x]+= x1*y1-x0*y0;
                pSumYY[x]+= y1*y1-y0*y0;
            }
        }
    }
};
float SSIM_Byte(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height, int win_size, int maxVal)
{
    WinSum3F<Byte> winSum(pDataX, pDataY, widthBytes, width, height, win_size, maxVal);
    return winSum.DoFilter();
}
float SSIM_Float(float* pDataX, float* pDataY, int step, int width, int height, int win_size, double maxVal)
{
    WinSum3F<float> winSum(pDataX, pDataY, step, width, height, win_size, maxVal);
    return winSum.DoFilter();
}

template <typename T, typename Data>
static inline void calcMeanValue(Data* pDataX, Data* pDataY, int widthBytes, int width, int height, int cx, int cy, int idx, int win_size,
    OUT T& meanX, OUT T& meanY, OUT T& sigmaXX, OUT T& sigmaXY, OUT T& sigmaYY)
{
    int cn=widthBytes/width, half_size=win_size/2;
    meanX=meanY=sigmaXX=sigmaXY=sigmaYY=0;
    for(int y=cy-half_size; y<=cy+half_size; y++)
    {
        Data *pX=pDataX+y*widthBytes;
        Data *pY=pDataY+y*widthBytes;
        for(int x=(cx-half_size)*cn+idx; x<(cx+half_size+1)*cn; x+=cn)
        {
            T fx=pX[x], fy=pY[x];
            meanX += fx;
            meanY += fy;
            sigmaXX += fx*fx;
            sigmaXY += fx*fy;
            sigmaYY += fy*fy;
        }
    }
    T invA=1/(T)(win_size*win_size);
    meanX=meanX*invA;  meanY=meanY*invA;
    sigmaXX=sigmaXX*invA;  sigmaXY=sigmaXY*invA;  sigmaYY=sigmaYY*invA;
}

template <typename T>
float SSIM_Byte_Slow_t(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height, int win_size)
{
    int half_size=win_size/2;
    win_size = 2*half_size+1;
    int count=0;
    double sum=0;
    T k1=0.01f, k2=0.03f, max_val=255, c1=SQR(k1*max_val), c2=SQR(k2*max_val);
    T conv_norm=(T)win_size*win_size/(win_size*win_size-1);
    int cn=widthBytes/width;
    for(int y=half_size; y<height-half_size; y++)
    {
        for(int k=0; k<cn; k++)
        {
            for(int x=half_size; x<width-half_size; x++)
            {
                T meanX, meanY, meanXX, meanXY, meanYY, sigmaXX, sigmaXY, sigmaYY;
                calcMeanValue(pDataX, pDataY, widthBytes, width, height, x, y, k, win_size, meanX, meanY, sigmaXX, sigmaXY, sigmaYY);
                meanXX = meanX*meanX;
                meanXY = meanX*meanY;
                meanYY = meanY*meanY;
                sigmaXX = conv_norm*(sigmaXX-meanXX);
                sigmaXY = conv_norm*(sigmaXY-meanXY);
                sigmaYY = conv_norm*(sigmaYY-meanYY);

                T u1=2*meanXY+c1, v1=2*sigmaXY+c2;
                T u2=meanXX+meanYY+c1, v2=sigmaXX+sigmaYY+c2;
                T z=(u1*v1)/(u2*v2);
                sum += z;
                count++;
            }
        }
    }
    return float(sum/count);
}
float SSIM_Byte_Slow(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height, int win_size)
{
    return SSIM_Byte_Slow_t<float>(pDataX, pDataY, widthBytes, width, height, win_size);
}


