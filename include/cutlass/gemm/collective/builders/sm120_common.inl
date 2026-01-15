/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


#pragma once

#include "cutlass/gemm/collective/builders/sm1xx_common.inl"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/arch/arch.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective::detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int sm120_smem_capacity_bytes = cutlass::arch::sm120_smem_capacity_bytes;
// Helper for selecting the shared memory copy atom to use for operand A
template <
  class ElementA,
  class ElementB,
  bool UseF8f6f4
>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_rr_smem_copy_selector_A() {
  if constexpr (UseF8f6f4) {
    if constexpr (sizeof_bits_v<ElementA> == 6) {
      return SM100_SU6_DU8x16_x4_LDSM_N{};
    }
    else if constexpr (sizeof_bits_v<ElementA> == 4) {
      return SM100_SU4_DU8x16_x4_LDSM_N{};
    }
    else {
      return SM75_U32x4_LDSM_N{};
    }
  }
  else {
    return SM75_U32x4_LDSM_N{};
  }
}

// Helper for selecting the shared memory copy atom to use for operand B
// The TileN parameter allows selecting appropriate copy atoms based on divisibility.
// x4 variants require TileN divisible by 32, x2 by 16, x1 by 8.
template <
  class ElementA,
  class ElementB,
  bool UseF8f6f4,
  int TileN = 128
>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_rr_smem_copy_selector_B() {
  if constexpr (UseF8f6f4) {
    if constexpr (sizeof_bits_v<ElementB> == 6) {
      // FP6: select based on TileN divisibility
      if constexpr (TileN % 32 == 0) {
        return SM100_SU6_DU8x16_x4_LDSM_N{};
      }
      else if constexpr (TileN % 16 == 0) {
        return SM100_SU6_DU8x16_x2_LDSM_N{};
      }
      else {
        return SM100_SU6_DU8x16_x1_LDSM_N{};
      }
    }
    else if constexpr (sizeof_bits_v<ElementB> == 4) {
      // FP4: select based on TileN divisibility
      if constexpr (TileN % 32 == 0) {
        return SM100_SU4_DU8x16_x4_LDSM_N{};
      }
      else if constexpr (TileN % 16 == 0) {
        return SM100_SU4_DU8x16_x2_LDSM_N{};
      }
      else {
        return SM100_SU4_DU8x16_x1_LDSM_N{};
      }
    }
    else {
      // FP8: use standard ldmatrix variants
      if constexpr (TileN % 32 == 0) {
        return SM75_U32x4_LDSM_N{};
      }
      else if constexpr (TileN % 16 == 0) {
        return SM75_U32x2_LDSM_N{};
      }
      else {
        return SM75_U32x1_LDSM_N{};
      }
    }
  } 
  else {
    // Non-F8F6F4: use standard ldmatrix variants
    if constexpr (TileN % 32 == 0) {
      return SM75_U32x4_LDSM_N{};
    }
    else if constexpr (TileN % 16 == 0) {
      return SM75_U32x2_LDSM_N{};
    }
    else {
      return SM75_U32x1_LDSM_N{};
    }
  }
}

template <class ElementType, class MajorSize>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_rr_smem_selector() {
  static_assert(cutlass::sizeof_bits<ElementType>::value <= 8, "Unsupported element size.");
 
  if constexpr      (MajorSize{} % size<1>(UMMA::Layout_K_SW128_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_SW128_Atom<ElementType>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW64_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_SW64_Atom<ElementType>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW32_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_SW32_Atom<ElementType>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_INTER_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_INTER_Atom<ElementType>{};
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementType>, "No shared memory copy atom can be selected.");
  }
}

template <class ElementType, class MajorSize, class Sparsity>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_rr_smem_selector_sparse() {
  static_assert(cutlass::sizeof_bits<ElementType>::value <= 8, "Unsupported element size.");

   if constexpr      (MajorSize{} % size<1>(UMMA::Layout_K_SW128_SpAtom<ElementType, Sparsity{}>{}) == 0) {
    return UMMA::Layout_K_SW128_SpAtom<ElementType, Sparsity{}>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW64_SpAtom<ElementType, Sparsity{}>{}) == 0) {
    return UMMA::Layout_K_SW64_SpAtom<ElementType, Sparsity{}>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW32_SpAtom<ElementType, Sparsity{}>{}) == 0) {
    return UMMA::Layout_K_SW32_SpAtom<ElementType, Sparsity{}>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_INTER_SpAtom<ElementType, Sparsity{}>{}) == 0) {
    return UMMA::Layout_K_INTER_SpAtom<ElementType, Sparsity{}>{};
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementType>, "No shared memory copy atom can be selected.");
  }
}

template <int SFVectorSize, int TileN = 32>
CUTLASS_HOST_DEVICE constexpr
auto
sm120_tile_n_permute_selector() {
  // Permute in the N mode to allow a warp to own all the elements needed for SF reduction.
  // The layout size must match the tile N dimension.
  // MMA atom N = 8, so we tile atoms to reach the desired N.
  //
  // TileN >= 32: Original layout, tiles 4 atoms (8*4=32)
  // TileN == 24: Tiles 3 atoms (8*3=24)
  // TileN == 16: Tiles 2 atoms (8*2=16)
  // TileN == 8:  Single atom (8*1=8)

  if constexpr (TileN >= 32) {
    // Original: Shape<_8,_2,_2> = 32 elements
    return cute::Layout<cute::Shape<_8,_2,_2>, cute::Stride<_1, _16,_8>>{};
  }
  else if constexpr (TileN == 24) {
    // Shape<_8,_3> = 24 elements (3 atoms)
    return cute::Layout<cute::Shape<_8,_3>, cute::Stride<_1, _8>>{};
  }
  else if constexpr (TileN == 16) {
    // Shape<_8,_2> = 16 elements (2 atoms)
    return cute::Layout<cute::Shape<_8,_2>, cute::Stride<_1, _8>>{};
  }
  else if constexpr (TileN == 8) {
    // Shape<_8> = 8 elements (1 atom)
    return cute::Layout<cute::Shape<_8>, cute::Stride<_1>>{};
  }
  else {
    static_assert(cutlass::detail::dependent_false<cute::C<TileN>>,
      "Unsupported TileN for SM120 collective builder. Must be multiple of 8.");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective::detail
