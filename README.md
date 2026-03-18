# 以下为 md 文件渲染语法，建议渲染后阅读，希望诸君运行指令时不要将渲染语法粘贴至终端运行。（不理解可粘贴至豆包解答）

# 执行

运行以下指令进行安装

```bash
chmod +x ./intall.sh
./intall.sh
```

生成的 whl 包在 <code>Cpp-Python/ultra-infer/python/dist</code>路径下

**如果发现运行 <code>./intall.sh</code> 包没有更新，请手动执行**

```bash
pip install --force-reinstall ./dist/ultra_infer_npu_python-1.1.1-cp310-cp310-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 

# 验证

执行以下指令，运行结果为 <code>Success inference: XXXXX</code> 代表安装包成功

```bash
cd ./test
python get_mat_test_rk.py
```

# 接入自定义C++接口

## 步骤一

在 <code>Cpp-Python/ultra-infer/ultra_infer/runtime</code> 路径下，建议新建一个文件夹 xxx，在该文件下实现您封装后的 C++ 接口。

## 步骤二

在 /home/orangepi/Cpp-Python/ultra-infer/ultra_infer/runtime/runtime.h 的 Runtime 类中定义一个方法，如 MyInterface()，以下为案例实现说明

```C++

在 runtime.h 中定义您的接口MyInterface。**注意需要引入您定义接口所需要的头文件**

#include "my_interface.h" //（仅示例作用，诸君勿复制）

struct ULTRAINFER_DECL Runtime {
public:
    //.....（代表省略符，诸君勿复制）

    void MyInterface(int xxx, const std::string xxx); //（仅示例作用，诸君勿复制）

    //.....
}

在 /home/orangepi/Cpp-Python/ultra-infer/ultra_infer/runtime/runtime.cc 路径下，实现您的接口 MyInterface

    void Runtime::MyInterface(int xxx, const std::string xxx){
        // 在这里调用您的自定义代码接口
        /*
                XXXXX
        */
    }

```

## 步骤三

在开始此步骤前，希望诸君对Linux 下 Cmake 语法有一定了解，这里不提供 Cmake 教学资料怕误导诸君。

```bash
# 1、在 Cpp-Python/ultra-infer/CMakeLists.txt 文件中，建议在 187 行中添加如下命令


file(GLOB_RECURSE MY_SRCS ${CMAKE_SOURCE_DIR}/ultra_infer/runtime/XXX/*.cpp  ${CMAKE_SOURCE_DIR}/ultra_infer/runtime/XXX/*.cc) # 其中XXX为您的C++接口实现文件

# 2、在该文件 123 行添加如下：
include_directories(${CMAKE_SOURCE_DIR}/ultra_infer/runtime/xxx/)  #其中xxx为您的自定义文件，如果该路径下有子文件夹建议追加 include_directories(${CMAKE_SOURCE_DIR}/ultra_infer/runtime/xxx/子文件夹/) 

# 3、在该文件中如下位置添加指令（诸君若无法定位，可以全局搜索 if(ENABLE_RKNPU2_BACKEND) 找到对应位置））

if(ENABLE_RKNPU2_BACKEND)

    list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_RKNPU2_SRCS} ${GSJ_SRCS} ${RK_SRCS}) #修改此行

    修改为 list(APPEND ALL_DEPLOY_SRCS ${DEPLOY_RKNPU2_SRCS} ${GSJ_SRCS} ${RK_SRCS} ${MY_SRCS}) # 其中 MY_SRCS 是您上述定义的变量
endif()

# 4、如果您的接口函数调用了第三方库，比如opencv此类。需执行以下指令：
include_directories(${XXX}/include) #其中xxx为您的第三方库的头文件路径

target_link_libraries(
    ${OpenCV_LIBS}
    ${XXXX_LIBS} # 其中xxx为您的第三方库的库路径
)
```

# 步骤四

在 /home/orangepi/Cpp-Python/ultra-infer/ultra_infer/pybind/runtime.cc 文件中 pybind11::class_<Runtime>(m, "Runtime") 类中定义方法。这里是将对外 python 接口和您定义的 C++ 接口建立映射。

下面举几个案例

```C++
void MyInterface(int xxx, const std::string xxx); ---->  .def("MyInterface_python",        
                                                             [](Runtime &self, int xxx, std::string xxx){
                                                                // 下面的代码均为C++代码格式
                                                                self.MyInterface(xxx,xxx); //这就是您封装的C++接口
                                                             }


int MyInterface(int xxx, const std::string xxx); ---->  .def("MyInterface_python",        
                                                             [](Runtime &self, int xxx, std::string xxx){
                                                                // 下面的代码均为C++代码格式
                                                                int result = self.MyInterface(xxx,xxx); //这就是您封装的C++接口
                                                                return result;
                                                             }
cv::Mat MyInterface(cv::Mat xxx, const std::string xxx); ---->  .def("MyInterface_python",        
                                                             [](Runtime &self, pybind11::array xxx, std::string xxx){
                                                                // 下面的代码均为C++代码格式
                                                                cv::Mat result = self.MyInterface(xxx,xxx); //这就是您封装的C++接口
                                                                return result;
                                                             }
特殊的地方在于 pybind11::array 和 cv::Mat 是映射关系，其余数据结构相同。
其中 MyInterface_python 可以任意定义name
```

# 步骤五

在 /home/orangepi/Cpp-Python/3.10/lib/python3.10/site-packages/ultra_infer/runtime.py 文件中的 Runtime 类添加方法

```python
以 cv::Mat MyInterface(cv::Mat xxx, const std::string xxx) 接口为例

class Runtime:
    # **********
        def MyInterface(self, data , url): # name 可自定义

            return self.MyInterface_python(data,url) #MyInterface_python为步骤四中您自定义的name
    # **********
```

# 步骤六 使用方法
以 cv::Mat MyInterface(cv::Mat xxx, const std::string xxx) 接口为例
```python
from ultra_infer import ModelFormat, Runtime, RuntimeOption


ui_option = RuntimeOption()
ui_option.use_rknpu2()
ui_option.set_model_path(
    "./source/lenet5_32.rknn", "", ModelFormat.RKNN
)
ui_runtime = Runtime(ui_option)
img = cv2.imread("/home/orangepi/Cpp-Python/pingdi.jpg", cv2.COLOR_BGR2RGB)
url = "success"
result = ui_runtime.MyInterface(img,url) #result 为numpy数组格式
```

# 总结

本人能力有限，文档中多有疏漏，诸君按步骤来可能会出现意外问题，多多理解。在黑夜中多花点时间即可解决。
