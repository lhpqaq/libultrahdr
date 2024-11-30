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

#include "ultrahdr/gainmapmath.h"
#include <riscv_vector.h>
#include <cassert>
#include <arm_neon.h>

namespace ultrahdr {

static inline vint16m8_t yConversion_neon(vuint8m8_t y, vuint16m8_t u, vuint16m8_t v,
                                          vuint16m8_t coeffs) {
  // vint32m4_t lo = __riscv_mul_vx_i32m4(u, vget_low_u16(coeffs));
  int32x4_t lo = vmull_laneq_s16(vget_low_s16(u), coeffs, 0);
  int32x4_t hi = vmull_laneq_s16(vget_high_s16(u), coeffs, 0);
  lo = vmlal_laneq_s16(lo, vget_low_s16(v), coeffs, 1);
  hi = vmlal_laneq_s16(hi, vget_high_s16(v), coeffs, 1);

  // Descale result to account for coefficients being scaled by 2^14.
  uint16x8_t y_output =
      vreinterpretq_u16_s16(vcombine_s16(vqrshrn_n_s32(lo, 14), vqrshrn_n_s32(hi, 14)));
  return vreinterpretq_s16_u16(vaddw_u8(y_output, y));
}

void transformYuv420_rvv(jr_uncompressed_ptr image, const int16_t* coeffs_ptr) {
  assert(image->width % 16 == 0);
  uint8_t* y0_ptr = static_cast<uint8_t*>(image->data);
  uint8_t* y1_ptr = y0_ptr + image->luma_stride;
  uint8_t* u_ptr = static_cast<uint8_t*>(image->chroma_data);
  uint8_t* v_ptr = u_ptr + image->chroma_stride * (image->height / 2);
  size_t vl;

  const vint16m8_t coeffs = __riscv_vle16_v_i16m8(coeffs_ptr, vl);
  size_t h = 0;
  do {
    size_t w = 0;
    do {
      vl = __riscv_vsetvl_e16m8((image->width / 2) - w);
      assert(vl %= 2 == 0);

      vuint8m8_t y0 = __riscv_vle8_v_u8m8(y0_ptr + w * 2, vl);
      vuint8m8_t y1 = __riscv_vle8_v_u8m8(y1_ptr + w * 2, vl);

      vuint8m4_t u8 = __riscv_vle8_v_u8m4(u_ptr + w, vl / 2);
      vuint8m4_t v8 = __riscv_vle8_v_u8m4(v_ptr + w, vl / 2);

      vuint16m8_t u16 = __riscv_vzext_vf2_u16m8(u8, vl / 2);
      vuint16m8_t v16 = __riscv_vzext_vf2_u16m8(v8, vl / 2);

      // 假设是小端序，0x0001 -> 0x01 0x00
      // 1,2,3,4 -> 1,0,2,0,3,0,4,0
      vuint8m8_t u1 = __riscv_vreinterpret_v_u16m8_u8m8(u16);
      vuint8m8_t v1 = __riscv_vreinterpret_v_u16m8_u8m8(v16);

      // 1,0,2,0,3,0,4,0 -> 0,1,0,2,0,3,0,4
      vuint8m8_t u2 = __riscv_vslide1up_vx_u8m8(u2, 0, vl);
      vuint8m8_t v2 = __riscv_vslide1up_vx_u8m8(v2, 0, vl);

      // 1,1,2,2,3,3,4,4
      vuint8m8_t u =  __riscv_vadd_vv_u8m8(u1, u2, vl);
      vuint8m8_t v =  __riscv_vadd_vv_u8m8(v1, v2, vl);

    }
  }
}

void transformYuv420_neon(jr_uncompressed_ptr image, const int16_t* coeffs_ptr) {
  // Implementation assumes image buffer is multiple of 16.
  assert(image->width % 16 == 0);
  uint8_t* y0_ptr = static_cast<uint8_t*>(image->data);
  uint8_t* y1_ptr = y0_ptr + image->luma_stride;
  uint8_t* u_ptr = static_cast<uint8_t*>(image->chroma_data);
  uint8_t* v_ptr = u_ptr + image->chroma_stride * (image->height / 2);
  size_t vl = 8;
  // const vint16m8_t coeffs = __riscv_vle16_v_i16m8(coeffs_ptr, vl);

  // 加载 8 个 16 位
   const int16x8_t coeffs = vld1q_s16(coeffs_ptr);
  // const vuint16m8_t uv_bias = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vmv_v_v_i16m8(-128, vl), vl);

  // int16 的 -128 转换成 8个i16 = v128
  // i16 to u16
   const uint16x8_t uv_bias = vreinterpretq_u16_s16(vdupq_n_s16(-128));
  size_t h = 0;
  do {
    size_t w = 0;
    do {
      // vuint8m8_t y0 = __riscv_vle8_v_u8m8(y0_ptr + w * 2, vl);

      // 16 个 8 位 uint  = v128
      uint8x16_t y0 = vld1q_u8(y0_ptr + w * 2);
      uint8x16_t y1 = vld1q_u8(y1_ptr + w * 2);

      // 8 个 8 位 uint = v64
      uint8x8_t u = vld1_u8(u_ptr + w);
      uint8x8_t v = vld1_u8(v_ptr + w);

      // 128 bias for UV given we are using libjpeg; see:
      // https://github.com/kornelski/libjpeg/blob/master/structure.doc

      // vaddw_u8: 8 位无符号整数相加, 应该会将 u8扩展成u16
      int16x8_t u_wide_s16 = vreinterpretq_s16_u16(vaddw_u8(uv_bias, u));  // -128 + u
      int16x8_t v_wide_s16 = vreinterpretq_s16_u16(vaddw_u8(uv_bias, v));  // -128 + v

      // 好像是复制一遍，1，2，3，4 变成 1，1，2，2，3，3，4，4
      const int16x8_t u_wide_lo = vzip1q_s16(u_wide_s16, u_wide_s16);
      const int16x8_t u_wide_hi = vzip2q_s16(u_wide_s16, u_wide_s16);
      const int16x8_t v_wide_lo = vzip1q_s16(v_wide_s16, v_wide_s16);
      const int16x8_t v_wide_hi = vzip2q_s16(v_wide_s16, v_wide_s16);

      const int16x8_t y0_lo = yConversion_neon(vget_low_u8(y0), u_wide_lo, v_wide_lo, coeffs);
      const int16x8_t y0_hi = yConversion_neon(vget_high_u8(y0), u_wide_hi, v_wide_hi, coeffs);
      const int16x8_t y1_lo = yConversion_neon(vget_low_u8(y1), u_wide_lo, v_wide_lo, coeffs);
      const int16x8_t y1_hi = yConversion_neon(vget_high_u8(y1), u_wide_hi, v_wide_hi, coeffs);

      const int16x8_t new_u = uConversion_neon(u_wide_s16, v_wide_s16, coeffs);
      const int16x8_t new_v = vConversion_neon(u_wide_s16, v_wide_s16, coeffs);

      // Narrow from 16-bit to 8-bit with saturation.
      const uint8x16_t y0_output = vcombine_u8(vqmovun_s16(y0_lo), vqmovun_s16(y0_hi));
      const uint8x16_t y1_output = vcombine_u8(vqmovun_s16(y1_lo), vqmovun_s16(y1_hi));
      const uint8x8_t u_output = vqmovun_s16(vaddq_s16(new_u, vdupq_n_s16(128)));
      const uint8x8_t v_output = vqmovun_s16(vaddq_s16(new_v, vdupq_n_s16(128)));

      vst1q_u8(y0_ptr + w * 2, y0_output);
      vst1q_u8(y1_ptr + w * 2, y1_output);
      vst1_u8(u_ptr + w, u_output);
      vst1_u8(v_ptr + w, v_output);

      w += 8;
    } while (w < image->width / 2);
    y0_ptr += image->luma_stride * 2;
    y1_ptr += image->luma_stride * 2;
    u_ptr += image->chroma_stride;
    v_ptr += image->chroma_stride;
  } while (++h < image->height / 2);
}

void convert_rgb_to_yuv_rvv(uhdr_raw_image_ext_t* dst, const uhdr_raw_image_t* src) {
  uint32_t* rgbData = static_cast<uint32_t*>(src->planes[UHDR_PLANE_PACKED]);
  unsigned int srcStride = src->stride[UHDR_PLANE_PACKED];

  uint8_t* yData = static_cast<uint8_t*>(dst->planes[UHDR_PLANE_Y]);
  uint8_t* uData = static_cast<uint8_t*>(dst->planes[UHDR_PLANE_U]);
  uint8_t* vData = static_cast<uint8_t*>(dst->planes[UHDR_PLANE_V]);

  size_t vl;

  for (size_t i = 0; i < dst->h; i++) {
    for (size_t j = 0; j < dst->w; j += vl) {
      vl = __riscv_vsetvl_e32m8(dst->w - j);

      vuint32m8_t vrgb = __riscv_vle32_v_u32m8(rgbData + srcStride * i + j, vl);
      vfloat32m8_t vr = __riscv_vreinterpret_v_u32m8_f32m8(__riscv_vand_vx_u32m8(vrgb, 0xff, vl));
      vfloat32m8_t vg = __riscv_vreinterpret_v_u32m8_f32m8(
          __riscv_vand_vx_u32m8(__riscv_vsrl_vx_u32m8(vrgb, 8, vl), 0xff, vl));
      vfloat32m8_t vb = __riscv_vreinterpret_v_u32m8_f32m8(
          __riscv_vand_vx_u32m8(__riscv_vsrl_vx_u32m8(vrgb, 16, vl), 0xff, vl));

      // Normalize to [0, 1] range
      vr = __riscv_vfdiv_vf_f32m8(vr, 255.0f, vl);
      vg = __riscv_vfdiv_vf_f32m8(vg, 255.0f, vl);
      vb = __riscv_vfdiv_vf_f32m8(vb, 255.0f, vl);

      vfloat32m8_t vy = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(vr, 255.0f, vl), 0.5f, vl);
      vfloat32m8_t vu = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(vg, 255.0f, vl), 128.5f, vl);
      vfloat32m8_t vv = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(vb, 255.0f, vl), 128.5f, vl);

      vy = __riscv_vfmin_vf_f32m8(vy, 0.0f, vl);
      vy = __riscv_vfmax_vf_f32m8(vy, 255.0f, vl);
      vu = __riscv_vfmin_vf_f32m8(vu, 0.0f, vl);
      vu = __riscv_vfmax_vf_f32m8(vu, 255.0f, vl);
      vv = __riscv_vfmin_vf_f32m8(vv, 0.0f, vl);
      vv = __riscv_vfmax_vf_f32m8(vv, 255.0f, vl);

      // Store the results
      vuint16m4_t vy_u16 = __riscv_vfncvt_rtz_xu_f_w_u16m4(vy, vl);
      vuint16m4_t vu_u16 = __riscv_vfncvt_rtz_xu_f_w_u16m4(vu, vl);
      vuint16m4_t vv_u16 = __riscv_vfncvt_rtz_xu_f_w_u16m4(vv, vl);
      vuint8m2_t vy_u8 = __riscv_vncvt_x_x_w_u8m2(vy_u16, vl);
      vuint8m2_t vu_u8 = __riscv_vncvt_x_x_w_u8m2(vu_u16, vl);
      vuint8m2_t vv_u8 = __riscv_vncvt_x_x_w_u8m2(vv_u16, vl);

      __riscv_vse8_v_u8m2(yData + dst->stride[UHDR_PLANE_Y] * i + j, vy_u8, vl);
      __riscv_vse8_v_u8m2(uData + dst->stride[UHDR_PLANE_U] * i + j, vu_u8, vl);
      __riscv_vse8_v_u8m2(vData + dst->stride[UHDR_PLANE_V] * i + j, vv_u8, vl);
    }
  }
}

std::unique_ptr<uhdr_raw_image_ext_t> convert_raw_input_to_ycbcr_rvv(uhdr_raw_image_t* src) {
  if (src->fmt == UHDR_IMG_FMT_32bppRGBA8888) {
    std::unique_ptr<uhdr_raw_image_ext_t> dst = nullptr;
    dst = std::make_unique<uhdr_raw_image_ext_t>(UHDR_IMG_FMT_24bppYCbCr444, src->cg, src->ct,
                                                 UHDR_CR_FULL_RANGE, src->w, src->h, 64);
    convert_rgb_to_yuv_rvv(dst.get(), src);
    return dst;
  }
  return nullptr;
}
}  // namespace ultrahdr
