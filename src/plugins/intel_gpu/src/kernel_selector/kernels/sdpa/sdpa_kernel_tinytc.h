// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_tinytc.h"
#include "sdpa_kernel_base.h"

namespace kernel_selector {
class SDPAKernelTinyTC : public KernelBaseTinyTC {
public:
    using Parent = KernelBaseTinyTC;
    SDPAKernelTinyTC() : KernelBaseTinyTC("sdpa_tinytc") {}
    virtual ~SDPAKernelTinyTC() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    enum ScalarID {
        ID_TARGET_SEQLEN = 0,
        ID_SOURCE_SEQLEN = 1,
        ID_NUM_HEADS = 2,
        ID_BATCH = 3,
        ID_Q_SEQ_STRIDE = 4,
        ID_Q_HEAD_STRIDE = 5,
        ID_Q_BATCH_STRIDE = 6,
        ID_K_SEQ_STRIDE = 7,
        ID_K_HEAD_STRIDE = 8,
        ID_K_BATCH_STRIDE = 9,
        ID_V_SEQ_STRIDE = 10,
        ID_V_HEAD_STRIDE = 11,
        ID_V_BATCH_STRIDE = 12,
        ID_O_SEQ_STRIDE = 13,
        ID_O_HEAD_STRIDE = 14,
        ID_O_BATCH_STRIDE = 15,
        NUMBER_OF_SCALAR_IDS = 16,
    };

    bool Validate(const Params& p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    CommonDispatchData SetDefault(const sdpa_params& params) const;
    JitConstants GetJitConstants(const sdpa_params& params) const;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {};
    }
    clKernelData get_kernel_data(const sdpa_params& params) const;
    inline auto block_size(std::int64_t head_size) const -> std::int64_t {
        return 512 / (head_size / 64);
    }
};
}  // namespace kernel_selector
