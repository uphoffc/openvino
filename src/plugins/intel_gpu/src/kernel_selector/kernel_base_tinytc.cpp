// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_base_tinytc.h"

#include <sstream>

namespace kernel_selector {

std::shared_ptr<KernelString> KernelBaseTinyTC::GetKernelString(const std::string& kernel_name,
                                                                const std::pair<std::string, std::string>& jit,
                                                                const std::string& entry_point) const {
    auto kernel_string = std::make_shared<KernelString>();
    auto codes = db.get(kernel_name);

    if (codes.size()) {
        kernel_string->str = codes[0];
        kernel_string->jit = jit.first;
        kernel_string->undefs = jit.second;
        kernel_string->entry_point = entry_point;
        kernel_string->batch_compilation = false;
        kernel_string->language = KernelLanguage::TINYTC;
    }

    return kernel_string;
}
std::pair<std::string, std::string> KernelBaseTinyTC::CreateJit(const JitConstants& constants) const {
    std::ostringstream defs;
    for (auto& definition : constants.GetDefinitions()) {
        auto& name = definition.first;
        auto& value = definition.second;
        defs << "$" << name << "=" << value << "\n";
    }
    return {std::move(defs).str(), ""};
}

void KernelBaseTinyTC::FillCLKernelData(clKernelData& kernel,
                                        const CommonDispatchData& dispatchData,
                                        const EngineInfo& engine_info,
                                        const std::string& kernelMapName,
                                        const std::pair<std::string, std::string>& jit,
                                        const std::string& entryPoint,
                                        bool is_dynamic) const {
    if (!is_dynamic && !kernel.skip_execution)
        KernelBase::CheckDispatchData(kernelMapName, dispatchData, engine_info.maxWorkGroupSize);
    kernel.code.kernelString = GetKernelString(kernelMapName, jit, entryPoint);
    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
}

}  // namespace kernel_selector
