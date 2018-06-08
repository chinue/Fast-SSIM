# -*- coding: utf-8 -*-
import numpy as np
import math
from base import *
import skimage.measure
import scipy
import scipy.misc
import ctypes
import re

ssim_dll_path = os.path.split(os.path.realpath(__file__))[0]
ssim_dll_name = 'ssim.dll' if(os.name=='nt') else 'ssim.so'

class Loader:
    
    if(os.path.exists(os.path.join(ssim_dll_path, ssim_dll_name))):
        print_debug("load '%s'"%(os.path.join(ssim_dll_path, ssim_dll_name)), textColor='golden')
        dll = np.ctypeslib.load_library('ssim', ssim_dll_path)
    else:
        print_debug("load '%s' FAILED"%(os.path.join(ssim_dll_path, ssim_dll_name)), textColor='red')

    type_dict = {'int':ctypes.c_int, 'float':ctypes.c_float, 'double':ctypes.c_double, 'void':None,
                    'int32':ctypes.c_int32, 'uint32':ctypes.c_uint32, 'int16':ctypes.c_int16, 'uint16':ctypes.c_uint16,
                    'int8':ctypes.c_int8, 'uint8':ctypes.c_uint8, 'byte':ctypes.c_uint8,
                    'char*':ctypes.c_char_p,
                    'float*':np.ctypeslib.ndpointer(dtype='float32', ndim=1, flags='CONTIGUOUS'),
                    'int*':np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='CONTIGUOUS'),
                    'byte*':np.ctypeslib.ndpointer(dtype='uint8', ndim=1, flags='CONTIGUOUS')}
    @staticmethod
    def get_function(res_type='float', func_name='PSNR_Byte', arg_types=['Byte*', 'int', 'int', 'int', 'Byte*']):
        func = Loader.dll.__getattr__(func_name)
        func.restype = Loader.type_dict[res_type]
        func.argtypes = [Loader.type_dict[str.lower(x).replace(' ', '')] for x in arg_types]
        return func

    @staticmethod
    def get_function2(c_define='DLL_API float PSNR_Byte(const Byte* pSrcData, int step, int width, int height, OUT Byte* pDstData);'):
        r = re.search(r'(\w+)\s+(\w+)\s*\((.+)\)', c_define)
        assert(r!=None)
        r = r.groups()
        print(r)
        arg_list = r[2].split(',')
        arg_types=[]
        for a in arg_list:
            a_list = a.split()
            if('*' in a_list[-1]):
                arg = a_list[-1].split('*')[0]+'*' if(a_list[-1][0]!='*') else a_list[-2]+'*'
            else:
                arg = a_list[-3]+'*' if(a_list[-2]=='*') else a_list[-2]
            arg_types.append(arg)
        print_debug(arg_types, textColor='magenta')
        #print_debug('res_type=%s, func_name=%s, arg_types=%s'%(r[0], r[1], str(arg_types)), textColor='yellow')
        return Loader.get_function(r[0], r[1], arg_types)

    @staticmethod
    def had_member(name='dll'):
        return (name in Loader.__dict__.keys())

class DLL:
    @staticmethod
    def had_function(name='PSNR_Byte'):
        return (name in DLL.__dict__.keys())
    
    if(Loader.had_member('dll')):
        # float PSNR_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal);
        PSNR_Byte = Loader.get_function('float', 'PSNR_Byte', ['Byte*', 'Byte*', 'int', 'int', 'int', 'int'])

        # float PSNR_Float(float* pDataX, float* pDataY, int step, int width, int height, double maxVal);
        PSNR_Float = Loader.get_function('float', 'PSNR_Float', ['float*', 'float*', 'int', 'int', 'int', 'double'])

        # float SSIM_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int win_size, int maxVal);
        SSIM_Byte = Loader.get_function('float', 'SSIM_Byte', ['Byte*', 'Byte*', 'int', 'int', 'int', 'int', 'int'])

        # float SSIM_Float(float* pDataX, float* pDataY, int step, int width, int height, int win_size, double maxVal);
        SSIM_Float = Loader.get_function('float', 'SSIM_Float', ['float*', 'float*', 'int', 'int', 'int', 'int', 'double'])

def PSNR(x, y, max_value=None):
    [h,w,c] = x.shape
    x = x.astype('float32') if(x.dtype=='float64') else x
    y = y.astype('float32') if(y.dtype=='float64') else y
    if(DLL.had_function('PSNR_Byte') and x.dtype=='uint8' and y.dtype=='uint8'):
        return DLL.PSNR_Byte(x.reshape([-1]), y.reshape([-1]), w*c, w, h, 255 if(max_value==None) else int(max_value))
    if(DLL.had_function('PSNR_Float') and x.dtype=='float32' and y.dtype=='float32'):
        return DLL.PSNR_Float(x.reshape([-1]), y.reshape([-1]), w*c, w, h, 255.0 if(max_value==None) else float(max_value))
    return skimage.measure.compare_psnr(x, y, max_value)

def PSNR_slow(x_image, y_image, max_value=255.0):
    return skimage.measure.compare_psnr(x_image, y_image, max_value)

def SSIM(x, y, max_value=None, win_size=7):
    [h,w,c] = x.shape
    x = x.astype('float32') if(x.dtype=='float64') else x
    y = y.astype('float32') if(y.dtype=='float64') else y
    if(DLL.had_function('SSIM_Byte') and x.dtype=='uint8' and y.dtype=='uint8'):
        return DLL.SSIM_Byte(x.reshape([-1]), y.reshape([-1]), w*c, w, h, win_size, 255 if(max_value==None) else int(max_value))
    if(DLL.had_function('SSIM_Float') and x.dtype=='float32' and y.dtype=='float32'):
        return DLL.SSIM_Float(x.reshape([-1]), y.reshape([-1]), w*c, w, h, win_size, 255.0 if(max_value==None) else float(max_value))
    return skimage.measure.compare_ssim(x,y, win_size=win_size, data_range=max_value, multichannel=(x.ndim>2))

def SSIM_slow(x_image, y_image, max_value=255.0, win_size=7, use_sample_covariance=True):
    x = np.asarray(x_image, np.float32)
    y = np.asarray(y_image, np.float32)
    return skimage.measure.compare_ssim(x,y, win_size=win_size, data_range=max_value, multichannel=(x.ndim>2))

if __name__ == '__main__':

    T = Timer()
    path = os.path.join(ssim_dll_path, 'jpg')
    x = scipy.misc.imread(os.path.join(path, '0.jpg'))
    y = scipy.misc.imread(os.path.join(path, '1.jpg'))

    #x = x.astype('float32')/255
    #y = y.astype('float32')/255
    print2(x.dtype)
    max_val = 255 if(x.dtype=='uint8') else 1.0

    T.begin()
    psnr = PSNR_slow(x, y, max_val)
    T.end("PSNR_slow")
    print2("  psnr_slow=%f"%psnr, textColor='cyan')

    T.begin()
    p = PSNR(x, y, max_val)
    T.end("PSNR_fast")
    print2("  psnr_fast=%f"%p, textColor='white')

    T.begin()
    ssim = SSIM_slow(x, y, max_val)
    T.end("SSIM_slow")
    print2("  ssim_slow=%f"%ssim, textColor='cyan')

    T.begin()
    p = SSIM(x, y, max_val)
    T.end("SSIM_fast")
    print2("  ssim_fast=%f"%p, textColor='white')

