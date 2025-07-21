// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "jitter.h"
#include "kernel_base.h"

namespace kernel_selector {

class KernelBaseTinyTC : public KernelBase {
public:
    using KernelBase::KernelBase;
    virtual ~KernelBaseTinyTC() {}

protected:
    virtual bool Validate(const Params&) const {
        return true;
    }
    std::shared_ptr<KernelString> GetKernelString(const std::string& kernel_name,
                                                  const std::pair<std::string, std::string>& jit,
                                                  const std::string& entry_point) const;
    std::pair<std::string, std::string> CreateJit(const JitConstants& constants) const;
    void FillCLKernelData(clKernelData& kernel,
                          const CommonDispatchData& dispatchData,
                          const EngineInfo& engine_info,
                          const std::string& kernelMapName,
                          const std::pair<std::string, std::string>& jit,
                          const std::string& entryPoint,
                          bool is_dynamic) const;
};
}  // namespace kernel_selector
