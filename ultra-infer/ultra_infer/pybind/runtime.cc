// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ultra_infer/pybind/main.h"

namespace ultra_infer {

void BindOption(pybind11::module &m);

void BindRuntime(pybind11::module &m) {
  BindOption(m);

  pybind11::class_<TensorInfo>(m, "TensorInfo")
      .def_readwrite("name", &TensorInfo::name)
      .def_readwrite("shape", &TensorInfo::shape)
      .def_readwrite("dtype", &TensorInfo::dtype);

  pybind11::class_<Runtime>(m, "Runtime")
      .def(pybind11::init())
      .def("init", &Runtime::Init)
      .def("compile",
           [](Runtime &self,
              std::vector<std::vector<pybind11::array>> &warm_datas,
              const RuntimeOption &_option) {
             size_t rows = warm_datas.size();
             size_t columns = warm_datas[0].size();
             std::vector<std::vector<FDTensor>> warm_tensors(
                 rows, std::vector<FDTensor>(columns));
             for (size_t i = 0; i < rows; ++i) {
               for (size_t j = 0; j < columns; ++j) {
                 auto dtype =
                     NumpyDataTypeToFDDataType(warm_datas[i][j].dtype());
                 std::vector<int64_t> data_shape;
                 data_shape.insert(data_shape.begin(), warm_datas[i][j].shape(),
                                   warm_datas[i][j].shape() +
                                       warm_datas[i][j].ndim());
                 warm_tensors[i][j].Resize(data_shape, dtype);
                 memcpy(warm_tensors[i][j].MutableData(),
                        warm_datas[i][j].mutable_data(),
                        warm_datas[i][j].nbytes());
               }
             }
             return self.Compile(warm_tensors);
           })
      .def("infer",
           [](Runtime &self, std::map<std::string, pybind11::array> &data) {
             std::vector<FDTensor> inputs(data.size());
             int index = 0;
             for (auto iter = data.begin(); iter != data.end(); ++iter) {
               std::vector<int64_t> data_shape;
               data_shape.insert(data_shape.begin(), iter->second.shape(),
                                 iter->second.shape() + iter->second.ndim());
               auto dtype = NumpyDataTypeToFDDataType(iter->second.dtype());
               // TODO(jiangjiajun) Maybe skip memory copy is a better choice
               // use SetExternalData
               inputs[index].Resize(data_shape, dtype);
               memcpy(inputs[index].MutableData(), iter->second.mutable_data(),
                      iter->second.nbytes());
               inputs[index].name = iter->first;
               index += 1;
             }

             std::vector<FDTensor> outputs(self.NumOutputs());
             self.Infer(inputs, &outputs);

             std::vector<pybind11::array> results;
             results.reserve(outputs.size());
             for (size_t i = 0; i < outputs.size(); ++i) {
               auto numpy_dtype = FDDataTypeToNumpyDataType(outputs[i].dtype);
               results.emplace_back(
                   pybind11::array(numpy_dtype, outputs[i].shape));
               memcpy(results[i].mutable_data(), outputs[i].Data(),
                      outputs[i].Numel() * FDDataTypeSize(outputs[i].dtype));
             }
             return results;
           })
      .def("infer_gsj",
           [](Runtime &self, std::map<std::string, pybind11::array> &data,int rows, int cols) {
             std::vector<FDTensor> inputs(data.size());
             int index = 0;
             for (auto iter = data.begin(); iter != data.end(); ++iter) {
               std::vector<int64_t> data_shape;
               data_shape.insert(data_shape.begin(), iter->second.shape(),
                                 iter->second.shape() + iter->second.ndim());
               auto dtype = NumpyDataTypeToFDDataType(iter->second.dtype());
               dtype= FDDataType::UINT8;
               // TODO(jiangjiajun) Maybe skip memory copy is a better choice
               // use SetExternalData
               inputs[index].Resize(data_shape, dtype);
               memcpy(inputs[index].MutableData(), iter->second.mutable_data(),
                      iter->second.nbytes());
               inputs[index].name = iter->first;
               index += 1;
             }      
            //  std::vector<FDTensor> outputs(self.NumOutputs());  
             std::vector<FDTensor> outputs(5);   //********** */
             self.InferGsj(inputs, &outputs,rows,cols);            
             std::vector<pybind11::array> results;
             results.reserve(outputs.size());
             for (size_t i = 0; i < outputs.size(); ++i) {
               auto numpy_dtype = FDDataTypeToNumpyDataType(outputs[i].dtype);
               results.emplace_back(
                   pybind11::array(numpy_dtype, outputs[i].shape));
               memcpy(results[i].mutable_data(), outputs[i].Data(),
                      outputs[i].Numel() * FDDataTypeSize(outputs[i].dtype));
             }
             return results;
           })      
      .def("get_mat",
           [](Runtime &self) {
             std::vector<FDTensor> outputs(1);   //********** */
             self.GetMat(&outputs);            
             std::vector<pybind11::array> results;
             results.reserve(outputs.size());
             for (size_t i = 0; i < outputs.size(); ++i) {
               auto numpy_dtype = FDDataTypeToNumpyDataType(outputs[i].dtype);
               results.emplace_back(
                   pybind11::array(numpy_dtype, outputs[i].shape));
               memcpy(results[i].mutable_data(), outputs[i].Data(),
                      outputs[i].Numel() * FDDataTypeSize(outputs[i].dtype));
             }
             return results;
           })   
      // .def("get_mat_rkmpp",
      //      [](Runtime &self, std::string &url_str) {
      //        std::vector<FDTensor> output_tensors;   //********** */
      //        std::vector<cv::Mat> outputs = self.GetMatRkMpp(url_str); 
      //        output_tensors.resize(outputs.size());
      //        for(int i = 0 ; i < outputs.size(); i++){
      //           size_t required_size = outputs[i].total() * outputs[i].elemSize();
      //           output_tensors[i].buffer_ = malloc(required_size);
      //           std::memcpy(output_tensors[i].buffer_, outputs[i].data, outputs[i].total() * outputs[i].elemSize()); 
      //           std::vector<int64_t> output_shape ;
      //           output_shape.push_back(outputs[i].rows);
      //           output_shape.push_back(outputs[i].cols);
      //           output_shape.push_back(outputs[i].channels());
      //           output_tensors[i].shape = output_shape;
      //           output_tensors[i].dtype = FDDataType::UINT8;
      //        }        
      //        std::vector<pybind11::array> results;
      //        results.reserve(output_tensors.size());
      //        for (size_t i = 0; i < output_tensors.size(); ++i) {
      //          auto numpy_dtype = FDDataTypeToNumpyDataType(output_tensors[i].dtype);
      //          results.emplace_back(
      //              pybind11::array(numpy_dtype, output_tensors[i].shape));
      //          memcpy(results[i].mutable_data(), output_tensors[i].Data(),
      //                 output_tensors[i].Numel() * FDDataTypeSize(output_tensors[i].dtype));
      //        }
      //        return results;
      //      })                   
      .def("init_mat",
           [](Runtime &self, std::string &url_str,std::string ip = "192.168.1.35",std::string password = "firefly") {
               self.InitMat(url_str,ip,password); 
           })          
      .def("init_mat_rkmpp",
           [](Runtime &self, std::string &url_str,std::string ip = "192.168.1.35",std::string password = "firefly") {
               self.InitMatRkmpp(url_str,ip,password); 
           })  
      .def("stop",
           [](Runtime &self) {
               self.Stop(); 
           })                                 
      .def("infer",
           [](Runtime &self, std::map<std::string, FDTensor> &data) {
             std::vector<FDTensor> inputs;
             inputs.reserve(data.size());
             for (auto iter = data.begin(); iter != data.end(); ++iter) {
               FDTensor tensor;
               tensor.SetExternalData(iter->second.Shape(),
                                      iter->second.Dtype(), iter->second.Data(),
                                      iter->second.device);
               tensor.name = iter->first;
               inputs.push_back(tensor);
             }
             std::vector<FDTensor> outputs;
             if (!self.Infer(inputs, &outputs)) {
               throw std::runtime_error("Failed to inference with Runtime.");
             }
             return outputs;
           })
      .def("infer",
           [](Runtime &self, std::vector<FDTensor> &inputs) {
             std::vector<FDTensor> outputs;
             self.Infer(inputs, &outputs);
             return outputs;
           })
      .def("bind_input_tensor", &Runtime::BindInputTensor)
      .def("bind_output_tensor", &Runtime::BindOutputTensor)
      .def("infer", [](Runtime &self) { self.Infer(); })
      .def("get_output_tensor",
           [](Runtime &self, const std::string &name) {
             FDTensor *output = self.GetOutputTensor(name);
             if (output == nullptr) {
               return pybind11::cast(nullptr);
             }
             return pybind11::cast(*output);
           })
      .def("num_inputs", &Runtime::NumInputs)
      .def("num_outputs", &Runtime::NumOutputs)
      .def("get_input_info", &Runtime::GetInputInfo)
      .def("get_output_info", &Runtime::GetOutputInfo)
      .def("get_profile_time", &Runtime::GetProfileTime)
      .def_readonly("option", &Runtime::option);

  pybind11::enum_<Backend>(m, "Backend", pybind11::arithmetic(),
                           "Backend for inference.")
      .value("UNKOWN", Backend::UNKNOWN)
      .value("ORT", Backend::ORT)
      .value("TRT", Backend::TRT)
      .value("POROS", Backend::POROS)
      .value("PDINFER", Backend::PDINFER)
      .value("RKNPU2", Backend::RKNPU2)
      .value("SOPHGOTPU", Backend::SOPHGOTPU)
      .value("TVM", Backend::TVM)
      .value("LITE", Backend::LITE)
      .value("OMONNPU", Backend::OMONNPU);
  pybind11::enum_<ModelFormat>(m, "ModelFormat", pybind11::arithmetic(),
                               "ModelFormat for inference.")
      .value("PADDLE", ModelFormat::PADDLE)
      .value("TORCHSCRIPT", ModelFormat::TORCHSCRIPT)
      .value("RKNN", ModelFormat::RKNN)
      .value("SOPHGO", ModelFormat::SOPHGO)
      .value("ONNX", ModelFormat::ONNX)
      .value("TVMFormat", ModelFormat::TVMFormat)
      .value("OM", ModelFormat::OM);
  pybind11::enum_<Device>(m, "Device", pybind11::arithmetic(),
                          "Device for inference.")
      .value("CPU", Device::CPU)
      .value("GPU", Device::GPU)
      .value("IPU", Device::IPU)
      .value("RKNPU", Device::RKNPU)
      .value("SOPHGOTPU", Device::SOPHGOTPUD);

  pybind11::enum_<FDDataType>(m, "FDDataType", pybind11::arithmetic(),
                              "Data type of UltraInfer.")
      .value("BOOL", FDDataType::BOOL)
      .value("INT8", FDDataType::INT8)
      .value("INT16", FDDataType::INT16)
      .value("INT32", FDDataType::INT32)
      .value("INT64", FDDataType::INT64)
      .value("FP16", FDDataType::FP16)
      .value("FP32", FDDataType::FP32)
      .value("FP64", FDDataType::FP64)
      .value("UINT8", FDDataType::UINT8);

  m.def("get_available_backends", []() { return GetAvailableBackends(); });
}

} // namespace ultra_infer
