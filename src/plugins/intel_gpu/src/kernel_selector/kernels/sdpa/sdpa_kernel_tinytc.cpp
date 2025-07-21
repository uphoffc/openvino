// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_tinytc.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include "common_tools.h"
#include "common_types.h"
#include "jitter.h"
#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "tensor_type.h"
#include "tinytc/types.h"

namespace kernel_selector {

ParamsKey SDPAKernelTinyTC::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

bool SDPAKernelTinyTC::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (p.GetType() != KernelType::SDPA)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const sdpa_params& params = static_cast<const sdpa_params&>(p);
    for (size_t i = 0; i < params.inputs.size(); i++) {
        if (params.inputs[i].Dimentions() != 4)
            DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (params.outputs[0].Dimentions() != 4)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.should_use_sdpa_opt)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.engineInfo.arch < gpu_arch::xe_hpc)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.indirect_axis != -1)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.input0_order[3] != 3 || params.input1_order[3] != 3 || params.input2_order[3] != 3)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.conf.k_head_size != params.conf.v_head_size)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.conf.k_head_size % 64 != 0)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.conf.k_head_size > 512)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    if (params.conf.is_paged_attention) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (params.conf.is_causal) {
        // \todo
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (params.conf.has_const_attn_mask_val) {
        // \todo
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (params.inputs.size() > 3) {
        // \todo
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

JitConstants SDPAKernelTinyTC::GetJitConstants(const sdpa_params& params) const {
    auto jit = JitConstants{};
    const auto& prim_params = dynamic_cast<const sdpa_params&>(params);

    const auto& Q = prim_params.inputs[0];
    const auto& K = prim_params.inputs[1];
    const auto& V = prim_params.inputs[2];

    const auto k_head_size = prim_params.conf.k_head_size;
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", k_head_size));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", block_size(k_head_size)));

    const auto default_scale = [&params]() {
        return 1.0 / std::sqrt(static_cast<double>(params.conf.k_head_size));
    };
    const auto static_scale_value = std::to_string(params.conf.has_const_scale_val ? params.conf.scale_val : default_scale());
    jit.AddConstant(MakeJitConstant("STATIC_SCALE_VALUE", static_scale_value));

    jit.AddConstant(MakeJitConstant("QUERY_T", toTinyTCType(Q.GetDType())));
    jit.AddConstant(MakeJitConstant("KEY_T", toTinyTCType(K.GetDType())));
    jit.AddConstant(MakeJitConstant("VALUE_T", toTinyTCType(V.GetDType())));

    return jit;
}

CommonDispatchData SDPAKernelTinyTC::SetDefault(const sdpa_params& params) const {
    CommonDispatchData dispatch_data;

    const auto& Q_modes = params.inputs[0].GetDims();
    const auto target_seqlen = Q_modes[params.input0_order[1]].v;
    const auto num_heads = Q_modes[params.input0_order[2]].v;
    const auto batch = Q_modes[params.input0_order[3]].v;

    const auto bs = block_size(params.conf.k_head_size);
    const auto num_blocks = 1 + (target_seqlen - 1) / bs;

    dispatch_data.lws[0] = 16;
    dispatch_data.lws[1] = 32;
    dispatch_data.lws[2] = 1;
    dispatch_data.gws = dispatch_data.lws;
    dispatch_data.gws[0] *= num_blocks;
    dispatch_data.gws[1] *= num_heads;
    dispatch_data.gws[2] *= batch;

    return dispatch_data;
}

clKernelData SDPAKernelTinyTC::get_kernel_data(const sdpa_params& params) const {
    auto dispatch_data = SetDefault(params);
    const auto& entry_point = "flash_attention";
    auto jit = CreateJit(GetJitConstants(params));
    clKernelData kernel;

    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point, params.is_shape_agnostic);

    ScalarDescriptor index_prototype;
    index_prototype.t = ScalarDescriptor::Types::INT64;
    index_prototype.v.s64 = 0;
    kernel.params.scalars = std::vector<ScalarDescriptor>(NUMBER_OF_SCALAR_IDS, index_prototype);

    kernel.params.arguments.clear();
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});                   // Q
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_TARGET_SEQLEN});   // target seqlen
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_NUM_HEADS});       // num_heads
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_BATCH});           // batch
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_Q_SEQ_STRIDE});    // Q seq stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_Q_HEAD_STRIDE});   // Q head stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_Q_BATCH_STRIDE});  // Q batch stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});                   // K
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_SOURCE_SEQLEN});   // source seqlen
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_NUM_HEADS});       // num_heads
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_BATCH});           // batch
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_K_SEQ_STRIDE});    // K seq stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_K_HEAD_STRIDE});   // K head stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_K_BATCH_STRIDE});  // K batch stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});                   // V
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_SOURCE_SEQLEN});   // source seqlen
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_NUM_HEADS});       // num_heads
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_BATCH});           // batch
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_V_SEQ_STRIDE});    // V seq stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_V_HEAD_STRIDE});   // V head stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_V_BATCH_STRIDE});  // V batch stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});                  // A
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_TARGET_SEQLEN});   // target seqlen
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_NUM_HEADS});       // num_heads
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_BATCH});           // batch
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_O_SEQ_STRIDE});    // O seq stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_O_HEAD_STRIDE});   // O head stride
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, ID_O_BATCH_STRIDE});  // O batch stride

    kernel.code.kernelString->flags |= tinytc_core_feature_flag_large_register_file;

    return kernel;
}

KernelsData SDPAKernelTinyTC::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const size_t num_kernels = 1;
    KernelData kd = KernelData::Default<sdpa_params>(params, num_kernels);

    const auto& prim_params = dynamic_cast<const sdpa_params&>(params);
    for (size_t i = 0; i < num_kernels; i++) {
        kd.kernels[i] = get_kernel_data(prim_params);
    }

    GetUpdateDispatchDataFunc(kd);

    return {kd};
}

void SDPAKernelTinyTC::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
        OPENVINO_ASSERT(kernel_data.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");

        const auto& prim_params = static_cast<const sdpa_params&>(params);
        auto& kd = kernel_data.kernels[0];

        auto dispatchData = SetDefault(prim_params);
        kd.params.workGroups.global = dispatchData.gws;
        kd.params.workGroups.local = dispatchData.lws;

        const auto& Q_modes = prim_params.inputs[0].GetDims();
        const auto& K_modes = prim_params.inputs[1].GetDims();
        const auto& V_modes = prim_params.inputs[2].GetDims();
        const auto& O_modes = prim_params.outputs[0].GetDims();

        kd.params.scalars[ID_TARGET_SEQLEN].v.s64 = Q_modes[prim_params.input0_order[1]].v;
        kd.params.scalars[ID_SOURCE_SEQLEN].v.s64 = K_modes[prim_params.input1_order[1]].v;
        kd.params.scalars[ID_NUM_HEADS].v.s64 = Q_modes[prim_params.input0_order[2]].v;
        kd.params.scalars[ID_BATCH].v.s64 = Q_modes[prim_params.input0_order[3]].v;
        kd.params.scalars[ID_Q_SEQ_STRIDE].v.s64 = Q_modes[prim_params.input0_order[1]].pitch;
        kd.params.scalars[ID_Q_HEAD_STRIDE].v.s64 = Q_modes[prim_params.input0_order[2]].pitch;
        kd.params.scalars[ID_Q_BATCH_STRIDE].v.s64 = Q_modes[prim_params.input0_order[3]].pitch;
        kd.params.scalars[ID_K_SEQ_STRIDE].v.s64 = K_modes[prim_params.input1_order[1]].pitch;
        kd.params.scalars[ID_K_HEAD_STRIDE].v.s64 = K_modes[prim_params.input1_order[2]].pitch;
        kd.params.scalars[ID_K_BATCH_STRIDE].v.s64 = K_modes[prim_params.input1_order[3]].pitch;
        kd.params.scalars[ID_V_SEQ_STRIDE].v.s64 = V_modes[prim_params.input2_order[1]].pitch;
        kd.params.scalars[ID_V_HEAD_STRIDE].v.s64 = V_modes[prim_params.input2_order[2]].pitch;
        kd.params.scalars[ID_V_BATCH_STRIDE].v.s64 = V_modes[prim_params.input2_order[3]].pitch;
        kd.params.scalars[ID_O_SEQ_STRIDE].v.s64 = O_modes[prim_params.output_order[1]].pitch;
        kd.params.scalars[ID_O_HEAD_STRIDE].v.s64 = O_modes[prim_params.output_order[2]].pitch;
        kd.params.scalars[ID_O_BATCH_STRIDE].v.s64 = O_modes[prim_params.output_order[3]].pitch;
    };
}

KernelsPriority SDPAKernelTinyTC::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
