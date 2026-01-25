source ~/3.10/bin/activate
cd ultra-infer/python
unset http_proxy https_proxy
# sudo rm -rf ./.setuptools-cmake-build ./ultra_infer_npu_python.egg-info ../third_party
export ENABLE_RKNPU2_BACKEND=ON ENABLE_ORT_BACKEND=ON ENABLE_OM_BACKEND=OFF  ENABLE_PADDLE_BACKEND=OFF WITH_GPU=OFF DEVICE_TYPE=NPU 
python setup.py build
pip install wheel
python setup.py bdist_wheel
pip install --force-reinstall ./dist/ultra_infer_npu_python-1.1.1-cp310-cp310-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple