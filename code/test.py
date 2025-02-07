from ctypes import CDLL
import os
dll_path = r"D:\\python\\Lib\\site-packages\\thundersvm\\thundersvm.dll"  # 替换为实际的 DLL 文件路
tem = CDLL(dll_path, winmode=0)

import thundersvm
print(thundersvm)
