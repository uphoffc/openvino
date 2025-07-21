; Copyright (C) 2025 Intel Corporation
; SPDX-License-Identifier: Apache-2.0

; %Q: (d,N,H,B)
; %K: (d,N,H,B)
; %V: (d,N,H,B)
; %O: (d,N,H,B)
func @flash_attention_d64(%Q: memref<f16x64x?x?x?,strided<1,?,?,?>> {stride_gcd=[1,64,64,64]},
                          %K: memref<f16x64x?x?x?,strided<1,?,?,?>> {stride_gcd=[1,64,64,64]},
                          %V: memref<f16x64x?x?x?,strided<1,?,?,?>> {stride_gcd=[1,64,64,64]},
                          %O: memref<f16x64x?x?x?,strided<1,?,?,?>> {stride_gcd=[1,64,64,64]})
                          attributes{subgroup_size=16, work_group_size=[16,32]} {
    %block_size = constant 512 : index
    %seq_len = size %K[1] : index
    %row_block = group_id.x : index
    %head = group_id.y : index
    %batch = group_id.z : index

    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %c32 = constant 32 : index
    %c64 = constant 64 : index
    %c512 = constant 512 : index
    %clog2e = constant 1.44269504088896340736 : f32
    %scale_factor = constant $STATIC_SCALE_VALUE : f32
    %scale = mul %scale_factor, %clog2e : f32

    %outer_offset = mul %row_block, %block_size : index
    %q = subview %Q[0:64,%outer_offset:512,%head,%batch] : memref<f16x64x512,strided<1,?>>
    %k = subview %K[0:64,%c0:%seq_len,%head,%batch] : memref<f16x64x?,strided<1,?>>
    %v = subview %V[0:64,%c0:%seq_len,%head,%batch] : memref<f16x64x?,strided<1,?>>
    %o = subview %O[0:64,%outer_offset:512,%head,%batch] : memref<f16x64x512,strided<1,?>>

    parallel {
        %o_init = constant 0.0 : coopmatrix<f32x32x16,matrix_acc>
        %s_init = constant 0.0 : coopmatrix<f32x32x16,matrix_acc>
        %maxvec_init = constant -inf : coopmatrix<f32x1x16,matrix_acc>
        %normvec_init = constant 0.0 : coopmatrix<f32x1x16,matrix_acc>
        %m_ones = constant -1.0 : coopmatrix<f32x32x1,matrix_a>
        %ones = constant 1.0 : coopmatrix<f32x32x1,matrix_a>

        %l_y = subgroup_id.y : i32
        %l_y_idx = cast %l_y : index
        %j0 = mul %c16, %l_y_idx : index

        for %j=%j0,%block_size,%c512 {
            %o_acc1,%o_acc2,%maxvec,%normvec = for %n=%c0,%seq_len,%c32
                init(%o_iter1=%o_init,%o_iter2=%o_init,
                     %maxvec_iter=%maxvec_init,%normvec_iter=%normvec_init)
                -> (coopmatrix<f32x32x16,matrix_acc>,coopmatrix<f32x32x16,matrix_acc>,
                    coopmatrix<f32x1x16,matrix_acc>,coopmatrix<f32x1x16,matrix_acc>) {
                cooperative_matrix_prefetch 0, %v[%c0,%n], 32, 32
                cooperative_matrix_prefetch 0, %v[%c32,%n], 32, 32

                %shat = for %m=%c0,%c64,%c32 init(%s_iter=%s_init) -> (coopmatrix<f32x32x16,matrix_acc>) {
                    %kk = cooperative_matrix_load.t.rows_checked %k[%m,%n] : coopmatrix<f16x32x32,matrix_a>
                    %qq = cooperative_matrix_load.cols_checked %q[%m,%j] : coopmatrix<f16x32x16,matrix_b>
                    %s_next = cooperative_matrix_mul_add %kk, %qq, %s_iter : coopmatrix<f32x32x16,matrix_acc>
                    yield (%s_next)
                } attributes{unroll=true}
                %s = cooperative_matrix_scale %scale, %shat : coopmatrix<f32x32x16,matrix_acc>

                %n_next = add %n, %c32 : index
                cooperative_matrix_prefetch 0, %k[%c0,%n_next], 64, 32

                %s_red = cooperative_matrix_reduce_max.column %s : coopmatrix<f32x1x16,matrix_acc>
                %maxvec_next = max %maxvec_iter, %s_red : coopmatrix<f32x1x16,matrix_acc>

                %maxvec_next_b = cast %maxvec_next : coopmatrix<f32x1x16,matrix_b>
                %s_diff = cooperative_matrix_mul_add %m_ones, %maxvec_next_b, %s : coopmatrix<f32x32x16,matrix_acc>
                %p = cooperative_matrix_apply (%x,%y,%val)=%s_diff -> coopmatrix<f32x32x16,matrix_acc> {
                    %expval = native_exp2 %val : f32
                    yield (%expval)
                }

                %maxvec_diff = sub %maxvec_iter, %maxvec_next : coopmatrix<f32x1x16,matrix_acc>
                %maxvec_diff_exp = cooperative_matrix_apply (%x,%y,%val)=%maxvec_diff
                                        -> coopmatrix<f32x1x16,matrix_acc> {
                    %expval = native_exp2 %val : f32
                    yield (%expval)
                }
                %normvec_rescaled = mul %maxvec_diff_exp, %normvec_iter : coopmatrix<f32x1x16,matrix_acc>
                %p_red = cooperative_matrix_reduce_add.column %p : coopmatrix<f32x1x16,matrix_acc>
                %normvec_next = add %normvec_rescaled, %p_red : coopmatrix<f32x1x16,matrix_acc>

                %maxvec_diff_exp_b = cast %maxvec_diff_exp : coopmatrix<f32x1x16,matrix_b>
                %maxvec_mat = cooperative_matrix_mul_add %ones, %maxvec_diff_exp_b, %o_init : coopmatrix<f32x32x16,matrix_acc>

                %p_b = cast %p : coopmatrix<f16x32x16,matrix_b>
                %vv1 = cooperative_matrix_load.cols_checked %v[%c0,%n] : coopmatrix<f16x32x32,matrix_a>
                %vv2 = cooperative_matrix_load.cols_checked %v[%c32,%n] : coopmatrix<f16x32x32,matrix_a>
                %o_rescaled1 = mul %maxvec_mat, %o_iter1 : coopmatrix<f32x32x16,matrix_acc>
                %o_next1 = cooperative_matrix_mul_add %vv1, %p_b, %o_rescaled1 : coopmatrix<f32x32x16,matrix_acc>
                %o_rescaled2 = mul %maxvec_mat, %o_iter2 : coopmatrix<f32x32x16,matrix_acc>
                %o_next2 = cooperative_matrix_mul_add %vv2, %p_b, %o_rescaled2 : coopmatrix<f32x32x16,matrix_acc>
                yield (%o_next1,%o_next2,%maxvec_next,%normvec_next)
            }
            %ones16 = constant 1.0 : coopmatrix<f32x1x16,matrix_acc>
            %normvec_inv = div %ones16, %normvec : coopmatrix<f32x1x16,matrix_acc>
            %normvec_inv_b = cast %normvec_inv : coopmatrix<f32x1x16,matrix_b>
            %normvec_inv_mat = cooperative_matrix_mul_add %ones, %normvec_inv_b, %o_init : coopmatrix<f32x32x16,matrix_acc>
            %o_scaled1 = mul %normvec_inv_mat, %o_acc1 : coopmatrix<f32x32x16,matrix_acc>
            %o_scaled2 = mul %normvec_inv_mat, %o_acc2 : coopmatrix<f32x32x16,matrix_acc>
            %o_scaled1_f16 = cast %o_scaled1 : coopmatrix<f16x32x16,matrix_acc>
            cooperative_matrix_store.cols_checked %o_scaled1_f16, %o[%c0,%j]
            %o_scaled2_f16 = cast %o_scaled2 : coopmatrix<f16x32x16,matrix_acc>
            cooperative_matrix_store.cols_checked %o_scaled2_f16, %o[%c32,%j]
        } attributes{unroll=false}
    }
}
