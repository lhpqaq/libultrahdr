/*
 * Copyright 2024 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ULTRAHDR_DSP_ARM_MEM_NEON_H
#define ULTRAHDR_DSP_ARM_MEM_NEON_H

#include <riscv_vector.h>

#include "ultrahdr/ultrahdrcommon.h"

namespace ultrahdr {

  int check_riscv_vector_support() {
      unsigned long misa;
      asm volatile ("csrr %0, misa" : "=r" (misa));
      
      if (misa & (1UL << 26)) {
          return 1;
      } else {
          return 0;
      }
  }
}  // namespace ultrahdr

#endif  // ULTRAHDR_DSP_RISCV_RVV_HELPER_H
