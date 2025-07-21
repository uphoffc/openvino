; Copyright (C) 2025 Intel Corporation
; SPDX-License-Identifier: Apache-2.0

$num_sg_x = !calc($HEAD_SIZE 32 /)
$num_sg_y = !calc($BLOCK_SIZE 32 /)
$sgs = 16
$wg_size_x = !calc($num_sg_x $sgs *)
$wg_size_y = $num_sg_y
$flash_attr = {subgroup_size=$sgs, work_group_size=[$wg_size_x,$wg_size_y]}

; %Q: (d,N,H,B)
; %K: (d,N,H,B)
; %V: (d,N,H,B)
; %O: (d,N,H,B)
func @flash_attention(%Q: memref<f16x$HEAD_SIZE x?x?x?,strided<1,?,?,?>> {stride_gcd=[1,64,64,64]},
                      %K: memref<f16x$HEAD_SIZE x?x?x?,strided<1,?,?,?>> {stride_gcd=[1,64,64,64]},
                      %V: memref<f16x$HEAD_SIZE x?x?x?,strided<1,?,?,?>> {stride_gcd=[1,64,64,64]},
                      %O: memref<f16x$HEAD_SIZE x?x?x?,strided<1,?,?,?>> {stride_gcd=[1,64,64,64]})
                      attributes $flash_attr {
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %c32 = constant 32 : index
    %c64 = constant 64 : index
    %c128 = constant 128 : index
    %c256 = constant 256 : index
    %cnum_subgroups_x = constant $num_sg_x : index
    %chead_size = constant $HEAD_SIZE : index
    %cblock_size = constant $BLOCK_SIZE : index

    %seq_len = size %K[1] : index
    %row_block = group_id.x : index
    %head = group_id.y : index
    %batch = group_id.z : index

    %clog2e = constant 1.44269504088896340736 : f32
    %scale_factor = constant $STATIC_SCALE_VALUE : f32
    %scale = mul %scale_factor, %clog2e : f32

    %outer_offset = mul %row_block, %cblock_size : index
    %q = subview %Q[0:$HEAD_SIZE,%outer_offset:%cblock_size,%head,%batch] : memref<f16x$HEAD_SIZE x?,strided<1,?>>
    %k = subview %K[0:$HEAD_SIZE,%c0:%seq_len,%head,%batch] : memref<f16x$HEAD_SIZE x?,strided<1,?>>
    %v = subview %V[0:$HEAD_SIZE,%c0:%seq_len,%head,%batch] : memref<f16x$HEAD_SIZE x?,strided<1,?>>
    %o = subview %O[0:$HEAD_SIZE,%outer_offset:%cblock_size,%head,%batch] : memref<f16x$HEAD_SIZE x?,strided<1,?>>

    %P = alloca {alignment=64} : memref<f16x32x32x $num_sg_x x $num_sg_y,local>
    %maxvec = alloca {alignment=64} : memref<f32x $BLOCK_SIZE x 1,local>
    %maxvec_diff_exp_tmp = alloca {alignment=64} : memref<f32x $BLOCK_SIZE x 1,local>
    %normvec = alloca {alignment=64} : memref<f32x $BLOCK_SIZE x 1,local>

    parallel {
        %o_init = constant 0.0 : coopmatrix<f32x32x32,matrix_acc>
        %s_init = constant 0.0 : coopmatrix<f32x32x32,matrix_acc>
        %maxvec_init = constant -inf : coopmatrix<f32x32x1,matrix_acc>
        %normvec_init = constant 0.0 : coopmatrix<f32x32x1,matrix_acc>
        %m_ones = constant -1.0 : coopmatrix<f32x1x32,matrix_b>
        %ones = constant 1.0 : coopmatrix<f32x32x1,matrix_a>

        %l_y = subgroup_id.y : i32
        %l_y_idx = cast %l_y : index
        %j0 = mul %c32, %l_y_idx : index

        %l_x = subgroup_id.x : i32
        %l_x_idx = cast %l_x : index
        %i0 = mul %c32, %l_x_idx : index
        %n0 = mul %c32, %l_x_idx : index

        %c0_i32 = constant 0 : i32
        %is_first_x = equal %l_x, %c0_i32 : bool
        if %is_first_x {
            cooperative_matrix_store %maxvec_init, %maxvec[%j0,%c0]
            cooperative_matrix_store %normvec_init, %normvec[%j0,%c0]
        }
        barrier.local

        %n_step = mul %c32, %cnum_subgroups_x : index

        %o_acc,%maxvec_acc = for %n=%c0,%seq_len,%n_step
            init(%o_iter=%o_init,%maxvec_iter=%maxvec_init)
            -> (coopmatrix<f32x32x32,matrix_acc>,coopmatrix<f32x32x1,matrix_acc>) {
            %n_kq = add %n, %n0 : index
            cooperative_matrix_prefetch 0, %v[%i0,%n_kq], 32, 32

            %shat = for %m=%c0,%chead_size,%c32 init(%s_iter=%s_init) -> (coopmatrix<f32x32x32,matrix_acc>) {
                %kk = cooperative_matrix_load.cols_checked %k[%m,%n_kq] : coopmatrix<f16x32x32,matrix_b>
                %qq = cooperative_matrix_load.t.rows_checked %q[%m,%j0] : coopmatrix<f16x32x32,matrix_a>
                %m_next = add %m, %c32 : index
                %s_next = cooperative_matrix_mul_add %qq, %kk, %s_iter : coopmatrix<f32x32x32,matrix_acc>
                yield (%s_next)
            } attributes{unroll=false}
            %s = cooperative_matrix_scale %scale, %shat : coopmatrix<f32x32x32,matrix_acc>

            ;%n_next = add %n_kq, %chead_size : index
            ;cooperative_matrix_prefetch 0, %k[%c0,%n_next], 128, 32

            %s_red = cooperative_matrix_reduce_max.row %s : coopmatrix<f32x32x1,matrix_acc>
            %s_red_up = cooperative_matrix_atomic_max %s_red, %maxvec[%j0,%c0] : coopmatrix<f32x32x1,matrix_acc> 
            barrier.local
            %maxvec_next = cooperative_matrix_load.n %maxvec[%j0,%c0] : coopmatrix<f32x32x1,matrix_acc>

            %maxvec_next_a = cast %maxvec_next : coopmatrix<f32x32x1,matrix_a>
            %s_diff = cooperative_matrix_mul_add %maxvec_next_a, %m_ones, %s : coopmatrix<f32x32x32,matrix_acc>
            %p = cooperative_matrix_apply (%x,%y,%val)=%s_diff -> coopmatrix<f32x32x32,matrix_acc> {
                %expval = native_exp2 %val : f32
                yield (%expval)
            }
            %p_f16 = cast %p : coopmatrix<f16x32x32,matrix_acc>
            %P_sub = subview %P[0:32,0:32,%l_x_idx,%l_y_idx] : memref<f16x32x32,local>
            cooperative_matrix_store.t %p_f16, %P_sub[%c0,%c0]

            %maxvec_diff = sub %maxvec_iter, %maxvec_next : coopmatrix<f32x32x1,matrix_acc>
            %maxvec_diff_exp = cooperative_matrix_apply (%x,%y,%val)=%maxvec_diff
                                    -> coopmatrix<f32x32x1,matrix_acc> {
                %expval = native_exp2 %val : f32
                yield (%expval)
            }
            cooperative_matrix_store %maxvec_diff_exp, %maxvec_diff_exp_tmp[%j0,%c0]

            if %is_first_x {
                %normvec_last = cooperative_matrix_load.n %normvec[%j0,%c0] : coopmatrix<f32x32x1,matrix_acc>
                %normvec_rescaled = mul %maxvec_diff_exp, %normvec_last : coopmatrix<f32x32x1,matrix_acc>
                cooperative_matrix_store %normvec_rescaled, %normvec[%j0,%c0]
            }
            %p_red = cooperative_matrix_reduce_add.row %p : coopmatrix<f32x32x1,matrix_acc>
            barrier.local
            %p_red_up = cooperative_matrix_atomic_add %p_red, %normvec[%j0,%c0] : coopmatrix<f32x32x1,matrix_acc>

            %o_update = for %m=%c0,%n_step,%c32
                                    init(%o_inner_iter=%o_init)
                                    -> (coopmatrix<f32x32x32,matrix_acc>) {
                %n_vp = add %n, %m : index
                %m_block = div %m, %c32 : index
                %P_sub_m = subview %P[0:32,0:32,%m_block,%l_y_idx] : memref<f16x32x32,local>

                %vv = cooperative_matrix_load.cols_checked %v[%i0,%n_vp] : coopmatrix<f16x32x32,matrix_a>
                %pp1 = cooperative_matrix_load %P_sub_m[%c0,%c0] : coopmatrix<f16x32x32,matrix_b>
                %o_inner_next = cooperative_matrix_mul_add %vv, %pp1, %o_inner_iter : coopmatrix<f32x32x32,matrix_acc>
                yield (%o_inner_next)
            } attributes{unroll=false}

            %maxvec_diff_exp_b = cooperative_matrix_load.t %maxvec_diff_exp_tmp[%j0,%c0] : coopmatrix<f32x1x32,matrix_b>
            %maxvec_mat = cooperative_matrix_mul_add %ones, %maxvec_diff_exp_b, %o_init : coopmatrix<f32x32x32,matrix_acc>
            %o_iter_rescaled = mul %maxvec_mat, %o_iter : coopmatrix<f32x32x32,matrix_acc>
            %o_next = add %o_iter_rescaled, %o_update : coopmatrix<f32x32x32,matrix_acc>

            yield (%o_next,%maxvec_next)
        } attributes{unroll=false}
        barrier.local
        %normvec_final = cooperative_matrix_load.t %normvec[%j0,%c0] : coopmatrix<f32x1x32,matrix_b>
        %ones32 = constant 1.0 : coopmatrix<f32x1x32,matrix_b>
        %normvec_inv = div %ones32, %normvec_final : coopmatrix<f32x1x32,matrix_b>
        %normvec_inv_mat = cooperative_matrix_mul_add %ones, %normvec_inv, %o_init : coopmatrix<f32x32x32,matrix_acc>
        %o_scaled = mul %normvec_inv_mat, %o_acc : coopmatrix<f32x32x32,matrix_acc>
        %o_scaled_f16 = cast %o_scaled : coopmatrix<f16x32x32,matrix_acc>
        cooperative_matrix_store.cols_checked %o_scaled_f16, %o[%i0,%j0]
    }
}
