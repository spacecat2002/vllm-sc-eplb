# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr

from ._ll_bf16_dotprod import LLBf16Dotprod


class LLBf16DualGate(LLBf16Dotprod):
    """Low-latency dual-gate GEMM with an independent next-gate delta."""

    @cute.jit
    def _accumulate_weight(
        self,
        acc: cute.Tensor,
        gA: cute.Tensor,
        gB: cute.Tensor,
        M: cutlass.Constexpr,
        tidx: cutlass.Int32,
        n_idx: cutlass.Int32,
    ):
        if const_expr(self.k_main_elems > 0):
            gA_main = self._make_k_slice(gA, 0, self.k_main_elems)
            gB_main = self._make_k_slice(gB, 0, self.k_main_elems)
            gA_vec = cute.logical_divide(gA_main, (None, self.main_vec_width))
            gB_vec = cute.logical_divide(gB_main, (None, self.main_vec_width))
            tA, tB = self._make_thread_vector_slice(
                gA_vec, gB_vec, tidx, n_idx, self.bs
            )
            self._vector_dotprod(acc, tA, tB, M, self.main_tiles, 16)

        if const_expr(self.k_tail_elems > 0):
            gA_tail = self._make_k_slice(gA, self.k_main_elems, self.k_tail_elems)
            gB_tail = self._make_k_slice(gB, self.k_main_elems, self.k_tail_elems)
            gA_vec = cute.logical_divide(gA_tail, (None, self.tail_vec_width))
            gB_vec = cute.logical_divide(gB_tail, (None, self.tail_vec_width))
            tA, tB = self._make_thread_vector_slice(
                gA_vec, gB_vec, tidx, n_idx, self.bs
            )
            self._vector_dotprod(acc, tA, tB, M, self.tail_tiles, 8)

        if const_expr(self.ks_full > 0):
            gA_scalar = self._make_k_slice(gA, self.k_done_all, self.k_scalar_full)
            gB_scalar = self._make_k_slice(gB, self.k_done_all, self.k_scalar_full)
            gA_vec = cute.logical_divide(gA_scalar, (None, 1))
            gB_vec = cute.logical_divide(gB_scalar, (None, 1))
            tA, tB = self._make_thread_vector_slice(
                gA_vec, gB_vec, tidx, n_idx, self.bs
            )
            self._vector_dotprod(acc, tA, tB, M, self.ks_full, 2)

        if const_expr(self.ks_part > 0):
            gA_part = self._make_k_slice(gA, self.k_part_offset, self.ks_part)
            gB_part = self._make_k_slice(gB, self.k_part_offset, self.ks_part)
            if tidx < self.ks_part:
                value = gB_part[n_idx, tidx].to(cutlass.Float32)
                for m in cutlass.range_constexpr(M):
                    acc[m] = acc[m] + gA_part[m, tidx].to(cutlass.Float32) * value

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gCurrent: cute.Tensor,
        gNext: cute.Tensor,
        gDelta: cute.Tensor,
        gCurrentOut: cute.Tensor,
        gNextOut: cute.Tensor,
        M: cutlass.Constexpr,
        current_experts: cutlass.Int32,
        next_experts: cutlass.Int32,
        stream: CUstream,
    ):
        self.kernel(
            gA,
            gCurrent,
            gNext,
            gDelta,
            gCurrentOut,
            gNextOut,
            M,
            current_experts,
        ).launch(
            grid=[current_experts + next_experts, 1, 1],
            block=[self.bs, 1, 1],
            smem=M * 4 * self.num_warps,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gCurrent: cute.Tensor,
        gNext: cute.Tensor,
        gDelta: cute.Tensor,
        gCurrentOut: cute.Tensor,
        gNextOut: cute.Tensor,
        M: cutlass.Constexpr,
        current_experts: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        output_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        acc = cute.make_rmem_tensor((M,), cutlass.Float32)
        acc.fill(0.0)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        is_current = output_idx < current_experts
        if is_current:
            self._accumulate_weight(acc, gA, gCurrent, M, tidx, output_idx)
        else:
            next_idx = output_idx - current_experts
            self._accumulate_weight(acc, gA, gNext, M, tidx, next_idx)
            self._accumulate_weight(acc, gA, gDelta, M, tidx, next_idx)

        for m in cutlass.range_constexpr(M):
            acc[m] = cute.arch.warp_reduction_sum(acc[m])

        smem_layout = cute.make_layout((M, self.num_warps), stride=(self.num_warps, 1))
        smem = cutlass.utils.SmemAllocator()
        partials_smem = smem.allocate_tensor(
            cutlass.Float32, smem_layout, byte_alignment=16
        )
        with cute.arch.elect_one():
            for m in cutlass.range_constexpr(M):
                partials_smem[m, warp_idx] = acc[m]

        cute.arch.sync_threads()
        if tidx == 0:
            for m in cutlass.range_constexpr(M):
                partials = partials_smem[m, None].load()
                result = partials.reduce(
                    cute.ReductionOp.ADD,
                    init_val=cutlass.Float32(0.0),
                    reduction_profile=0,
                )
                if is_current:
                    gCurrentOut[m, output_idx] = result
                else:
                    gNextOut[m, output_idx - current_experts] = result

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
