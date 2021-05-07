import Cython.Build
import distutils.core
import os

# ext = Cython.Build.cythonize('network.py')  # 传入文件
# distutils.core.setup(ext_modules=ext)

root = r'D:\ailia-paper-projects\water_extracting_project'
# root = r'D:\ailia-paper-projects\water_extracting_project\baseline'
list_dirs = os.listdir(root)
for file in list_dirs:
    if file.endswith('.py'):
        if file in ('py-to-pyd.py', 'exa_pyhon.py', 'main.py', '__init__.py'):
            continue
        _ext = Cython.Build.cythonize(os.path.join(root, file))
        distutils.core.setup(ext_modules=_ext)
print('finished!')