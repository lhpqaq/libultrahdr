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
// #include <arm_neon.h>

namespace ultrahdr {

// Scale all coefficients by 2^14 to avoid needing floating-point arithmetic. This can cause an off
 // by one error compared to the scalar floating-point implementation.

 // Removing conversion coefficients 1 and 0 from the group for each standard leaves 6 coefficients.
 // Pack them into a single 128-bit vector as follows, zeroing the remaining elements:
 // {Y1, Y2, U1, U2, V1, V2, 0, 0}

 // Yuv Bt709 -> Yuv Bt601
 // Y' = (1.0f * Y) + ( 0.101579f * U) + ( 0.196076f * V)
 // U' = (0.0f * Y) + ( 0.989854f * U) + (-0.110653f * V)
 // V' = (0.0f * Y) + (-0.072453f * U) + ( 0.983398f * V)
 __attribute__((aligned(16)))
 const int16_t kYuv709To601_coeffs_rvv[8] = {1664, 3213, 16218, -1813, -1187, 16112, 0, 0};

 // Yuv Bt709 -> Yuv Bt2100
 // Y' = (1.0f * Y) + (-0.016969f * U) + ( 0.096312f * V)
 // U' = (0.0f * Y) + ( 0.995306f * U) + (-0.051192f * V)
 // V' = (0.0f * Y) + ( 0.011507f * U) + ( 1.002637f * V)
 __attribute__((aligned(16)))
 const int16_t kYuv709To2100_coeffs_rvv[8] = {-278, 1578, 16307, -839, 189, 16427, 0, 0};

 // Yuv Bt601 -> Yuv Bt709
 // Y' = (1.0f * Y) + (-0.118188f * U) + (-0.212685f * V),
 // U' = (0.0f * Y) + ( 1.018640f * U) + ( 0.114618f * V),
 // V' = (0.0f * Y) + ( 0.075049f * U) + ( 1.025327f * V);
 __attribute__((aligned(16)))
 const int16_t kYuv601To709_coeffs_rvv[8] = {-1936, -3485, 16689, 1878, 1230, 16799, 0, 0};

 // Yuv Bt601 -> Yuv Bt2100
 // Y' = (1.0f * Y) + (-0.128245f * U) + (-0.115879f * V)
 // U' = (0.0f * Y) + ( 1.010016f * U) + ( 0.061592f * V)
 // V' = (0.0f * Y) + ( 0.086969f * U) + ( 1.029350f * V)
 __attribute__((aligned(16)))
 const int16_t kYuv601To2100_coeffs_rvv[8] = {-2101, -1899, 16548, 1009, 1425, 16865, 0, 0};

 // Yuv Bt2100 -> Yuv Bt709
 // Y' = (1.0f * Y) + ( 0.018149f * U) + (-0.095132f * V)
 // U' = (0.0f * Y) + ( 1.004123f * U) + ( 0.051267f * V)
 // V' = (0.0f * Y) + (-0.011524f * U) + ( 0.996782f * V)
 __attribute__((aligned(16)))
 const int16_t kYuv2100To709_coeffs_rvv[8] = {297, -1559, 16452, 840, -189, 16331, 0, 0};

 // Yuv Bt2100 -> Yuv Bt601
 // Y' = (1.0f * Y) + ( 0.117887f * U) + ( 0.105521f * V)
 // U' = (0.0f * Y) + ( 0.995211f * U) + (-0.059549f * V)
 // V' = (0.0f * Y) + (-0.084085f * U) + ( 0.976518f * V)
 __attribute__((aligned(16)))
 const int16_t kYuv2100To601_coeffs_rvv[8] = {1931, 1729, 16306, -976, -1378, 15999, 0, 0};

static inline vuint16m8_t zip_self(vuint16m4_t a, size_t vl) {
  // vuint16m4_t a_lo = __riscv_vget_v_u16m8_u16m4(a, 0);
  vuint32m8_t a_wide = __riscv_vzext_vf2_u32m8(a, vl / 4);
  vuint16m8_t a_zero = __riscv_vreinterpret_v_u32m8_u16m8(a_wide);
  vuint16m8_t a_zero_slide = __riscv_vslide1up_vx_u16m8(a_zero, 0, vl / 2);
  vuint16m8_t a_zip = __riscv_vadd_vv_u16m8(a_zero, a_zero_slide, vl / 2);
  return a_zip;
}

// static inline vuint16m8_t zip_self_high(vuint16m8_t a_hi, size_t vl) {
//   // vuint16m4_t a_lo = __riscv_vget_v_u16m8_u16m4(a_hi, vl / 2);
//   vuint32m8_t a_hi_wide = __riscv_vzext_vf2_u32m8(a_lo, vl / 4);
//   vuint16m8_t a_zero = __riscv_vreinterpret_v_u32m8_u16m8(a_hi_wide);
//   vuint16m8_t a_zero_slide = __riscv_vslide1up_vx_u16m8(a_zero, 0, vl / 2);
//   vuint16m8_t a_zip = __riscv_vadd_vv_u16m8(a_zero, a_zero_slide, vl / 2);
//   return a_zip;
// }

// static inline vint16m8_t yConversion_neon(vuint8m8_t y, vuint16m8_t u, vuint16m8_t v,
//                                           vuint16m8_t coeffs) {
//   // vint32m4_t lo = __riscv_mul_vx_i32m4(u, vget_low_u16(coeffs));
//   int32x4_t lo = vmull_laneq_s16(vget_low_s16(u), coeffs, 0);
//   int32x4_t hi = vmull_laneq_s16(vget_high_s16(u), coeffs, 0);
//   lo = vmlal_laneq_s16(lo, vget_low_s16(v), coeffs, 1);
//   hi = vmlal_laneq_s16(hi, vget_high_s16(v), coeffs, 1);

//   // Descale result to account for coefficients being scaled by 2^14.
//   uint16x8_t y_output =
//       vreinterpretq_u16_s16(vcombine_s16(vqrshrn_n_s32(lo, 14), vqrshrn_n_s32(hi, 14)));
//   return vreinterpretq_s16_u16(vaddw_u8(y_output, y));
// }
static inline vint16m4_t vqrshrn_n_s32(vint32m8_t a, const int b, size_t vl) {
  return __riscv_vnclip_wx_i16m4(a, b, vl);
}

static inline vuint8m4_t vget_low_u8(vuint8m8_t u) {
  return __riscv_vget_v_u8m8_u8m4(u, 0);
}

static inline vuint8m4_t vget_high_u8(vuint8m8_t u,  size_t vl) {
  return __riscv_vget_v_u8m8_u8m4(__riscv_vslidedown_vx_u8m8(u, vl / 2, vl), 0);
}

static inline vint16m4_t vget_low_s16(vint16m8_t u) {
  return __riscv_vget_v_i16m8_i16m4(u, 0);
}

static inline vint16m4_t vget_high_s16(vint16m8_t u,  size_t vl) {
  return __riscv_vget_v_i16m8_i16m4(__riscv_vslidedown_vx_i16m8(u, vl / 2, vl), 0);
}

static inline vuint16m4_t vget_low_u16(vuint16m8_t u) {
  return __riscv_vget_v_u16m8_u16m4(u, 0);
}

static inline vuint16m4_t vget_high_u16(vuint16m8_t u,  size_t vl) {
  return __riscv_vget_v_u16m8_u16m4(__riscv_vslidedown_vx_u16m8(u, vl / 2, vl), 0);
}

static inline vint16m8_t vcombine_s16(vint16m4_t a, vint16m4_t b, size_t vl) {
  vint16m8_t a_wide = __riscv_vlmul_ext_v_i16m4_i16m8(a);
  vint16m8_t b_wide = __riscv_vlmul_ext_v_i16m4_i16m8(b);
  return __riscv_vslideup_vx_i16m8(a_wide, b_wide, vl / 2, vl);
}

static inline vuint8m8_t vcombine_u8(vuint8m4_t a, vuint8m4_t b, size_t vl) {
  vuint8m8_t a_wide = __riscv_vlmul_ext_v_u8m4_u8m8(a);
  vuint8m8_t b_wide = __riscv_vlmul_ext_v_u8m4_u8m8(b);
  return __riscv_vslideup_vx_u8m8(a_wide, b_wide, vl / 2, vl);
}

static inline vint16m8_t yConversion_rvv(vuint8m4_t y, vint16m8_t u, vint16m8_t v,
                                          const int16_t* coeffs, size_t vl) {
  vint32m8_t u_lo = __riscv_vwmul_vx_i32m8(vget_low_s16(u), coeffs[0], vl / 2);
  vint32m8_t u_hi = __riscv_vwmul_vx_i32m8(vget_high_s16(u, vl), coeffs[0], vl / 2);

  vint32m8_t v_lo = __riscv_vwmul_vx_i32m8(vget_low_s16(v), coeffs[1], vl / 2);
  vint32m8_t v_hi = __riscv_vwmul_vx_i32m8(vget_high_s16(v, vl), coeffs[1], vl / 2);

  vint32m8_t lo = __riscv_vadd_vv_i32m8(u_lo, v_lo, vl / 2);
  vint32m8_t hi = __riscv_vadd_vv_i32m8(u_hi, v_hi, vl / 2);

  vint16m4_t lo_shr = vqrshrn_n_s32(lo, 14, vl / 2); 
  vint16m4_t hi_shr = vqrshrn_n_s32(hi, 14, vl / 2);

  vint16m8_t y_output = vcombine_s16(lo_shr, hi_shr, vl);
  vuint16m8_t y_u16 = __riscv_vreinterpret_v_i16m8_u16m8(y_output);
  vuint16m8_t y_ret = __riscv_vwaddu_wv_u16m8(y_u16, y, vl);
  return __riscv_vreinterpret_v_u16m8_i16m8(y_ret);
}

static inline vint16m8_t uConversion_rvv(vint16m8_t u, vint16m8_t v,
                                          const int16_t* coeffs, size_t vl) {
  vint32m8_t u_lo = __riscv_vwmul_vx_i32m8(vget_low_s16(u), coeffs[2], vl / 2);
  vint32m8_t u_hi = __riscv_vwmul_vx_i32m8(vget_high_s16(u, vl), coeffs[2], vl / 2);

  vint32m8_t v_lo = __riscv_vwmul_vx_i32m8(vget_low_s16(v), coeffs[3], vl / 2);
  vint32m8_t v_hi = __riscv_vwmul_vx_i32m8(vget_high_s16(v, vl), coeffs[3], vl / 2);

  vint32m8_t lo = __riscv_vadd_vv_i32m8(u_lo, v_lo, vl / 2);
  vint32m8_t hi = __riscv_vadd_vv_i32m8(u_hi, v_hi, vl / 2);

  vint16m4_t lo_shr = vqrshrn_n_s32(lo, 14, vl / 2); 
  vint16m4_t hi_shr = vqrshrn_n_s32(hi, 14, vl / 2);

  vint16m8_t u_output = vcombine_s16(lo_shr, hi_shr, vl);
  return u_output;
}

static inline vint16m8_t vConversion_rvv(vint16m8_t u, vint16m8_t v,
                                          const int16_t* coeffs, size_t vl) {
  vint32m8_t u_lo = __riscv_vwmul_vx_i32m8(vget_low_s16(u), coeffs[4], vl / 2);
  vint32m8_t u_hi = __riscv_vwmul_vx_i32m8(vget_high_s16(u, vl), coeffs[4], vl / 2);

  vint32m8_t v_lo = __riscv_vwmul_vx_i32m8(vget_low_s16(v), coeffs[5], vl / 2);
  vint32m8_t v_hi = __riscv_vwmul_vx_i32m8(vget_high_s16(v, vl), coeffs[5], vl / 2);

  vint32m8_t lo = __riscv_vadd_vv_i32m8(u_lo, v_lo, vl / 2);
  vint32m8_t hi = __riscv_vadd_vv_i32m8(u_hi, v_hi, vl / 2);

  vint16m4_t lo_shr = vqrshrn_n_s32(lo, 14, vl / 2); 
  vint16m4_t hi_shr = vqrshrn_n_s32(hi, 14, vl / 2);

  vint16m8_t v_output = vcombine_s16(lo_shr, hi_shr, vl);
  return v_output;
}

static inline vuint8m4_t vqmovun_s16(vint16m8_t a, size_t vl) {
  vuint16m8_t a_non_neg = __riscv_vreinterpret_v_i16m8_u16m8(__riscv_vmax_vx_i16m8(a, 0, vl));
  return __riscv_vnclipu_wx_u8m4(a_non_neg, 0, vl);
}

void transformYuv420_rvv(jr_uncompressed_ptr image, const int16_t* coeffs_ptr) {
  assert(image->width % 16 == 0);
  uint8_t* y0_ptr = static_cast<uint8_t*>(image->data);
  uint8_t* y1_ptr = y0_ptr + image->luma_stride;
  uint8_t* u_ptr = static_cast<uint8_t*>(image->chroma_data);
  uint8_t* v_ptr = u_ptr + image->chroma_stride * (image->height / 2);
  size_t vl;

  // const vint16m8_t coeffs = __riscv_vle16_v_i16m8(coeffs_ptr, vl);
  // const 
  size_t h = 0;
  do {
    size_t w = 0;
    do {
      // vl = __riscv_vsetvl_e16m8((image->width / 2) - w);
      vl = __riscv_vsetvl_e8m8((image->width / 2) - w);
      assert((vl %= 4) == 0);

      vuint8m8_t y0 = __riscv_vle8_v_u8m8(y0_ptr + w * 2, vl);
      vuint8m8_t y1 = __riscv_vle8_v_u8m8(y1_ptr + w * 2, vl);

      vuint8m4_t u8 = __riscv_vle8_v_u8m4(u_ptr + w, vl / 2);
      vuint8m4_t v8 = __riscv_vle8_v_u8m4(v_ptr + w, vl / 2);

      vuint16m8_t u16_wide = __riscv_vwaddu_vx_u16m8(u8, -128, vl / 2);
      vuint16m8_t v16_wide = __riscv_vwaddu_vx_u16m8(v8, -128, vl / 2);

      // vget_high_s16(u16_wide, vl / 2) 1/4vl
      // uu_wide_lo 1/2vl
      vuint16m8_t uu_wide_lo = zip_self(__riscv_vget_v_u16m8_u16m4(u16_wide, 0), vl);
      vuint16m8_t uu_wide_hi = zip_self(vget_high_u16(u16_wide, vl / 2), vl);
      vuint16m8_t uv_wide_lo = zip_self(__riscv_vget_v_u16m8_u16m4(v16_wide, 0), vl);
      vuint16m8_t uv_wide_hi = zip_self(vget_high_u16(v16_wide, vl / 2), vl);
      
      vint16m8_t u_wide_lo = __riscv_vreinterpret_v_u16m8_i16m8(uu_wide_lo);
      vint16m8_t v_wide_lo = __riscv_vreinterpret_v_u16m8_i16m8(uv_wide_lo);
      vint16m8_t u_wide_hi = __riscv_vreinterpret_v_u16m8_i16m8(uu_wide_hi);
      vint16m8_t v_wide_hi = __riscv_vreinterpret_v_u16m8_i16m8(uv_wide_hi);

      vint16m8_t y0_lo = yConversion_rvv(vget_low_u8(y0), u_wide_lo, v_wide_lo, coeffs_ptr, vl / 2);
      vint16m8_t y1_lo = yConversion_rvv(vget_low_u8(y1), u_wide_lo, v_wide_lo, coeffs_ptr, vl / 2);
      vint16m8_t y0_hi = yConversion_rvv(vget_high_u8(y0, vl / 2), u_wide_hi, v_wide_hi, coeffs_ptr, vl / 2);
      vint16m8_t y1_hi = yConversion_rvv(vget_high_u8(y1, vl / 2), u_wide_hi, v_wide_hi, coeffs_ptr, vl / 2);
      
      vint16m8_t u_wide_s16 = __riscv_vreinterpret_v_u16m8_i16m8(u16_wide);
      vint16m8_t v_wide_s16 = __riscv_vreinterpret_v_u16m8_i16m8(v16_wide);
      vint16m8_t new_u = uConversion_rvv(u_wide_s16, v_wide_s16, coeffs_ptr, vl / 2);
      vint16m8_t new_v = vConversion_rvv(u_wide_s16, v_wide_s16, coeffs_ptr, vl / 2);

      vuint8m8_t y0_output = vcombine_u8(vqmovun_s16(y0_lo, vl / 2), vqmovun_s16(y0_hi, vl / 2), vl);
      vuint8m8_t y1_output = vcombine_u8(vqmovun_s16(y1_lo, vl / 2), vqmovun_s16(y1_hi, vl / 2), vl);
      vuint8m4_t u_output = vqmovun_s16(__riscv_vadd_vx_i16m8(new_u, 128, vl / 2), vl / 2);
      vuint8m4_t v_output = vqmovun_s16(__riscv_vadd_vx_i16m8(new_v, 128, vl / 2), vl / 2);

      __riscv_vse8_v_u8m8(y0_ptr + w * 2, y0_output, vl);
      __riscv_vse8_v_u8m8(y1_ptr + w * 2, y1_output, vl);
      __riscv_vse8_v_u8m4(u_ptr + w, u_output, vl / 2);
      __riscv_vse8_v_u8m4(v_ptr + w, v_output, vl / 2);

      // here
      // vuint16m8_t u16 = __riscv_vzext_vf2_u16m8(u8, vl / 2);
      // vuint16m8_t v16 = __riscv_vzext_vf2_u16m8(v8, vl / 2);

      // 假设是小端序，0x0001 -> 0x01 0x00
      // 1,2,3,4 -> 1,-1,2,-1,3,-1,4,-1
      // vint8m8_t u1 = __riscv_vreinterpret_v_i16m8_i8m8(u16_wide);
      // vint8m8_t v1 = __riscv_vreinterpret_v_i16m8_i8m8(v16_wide);

      // // 1,-1,2,-1,3,-1,4,-1 -> -1,1,-1,2,-1,3,-1,4
      // vint8m8_t u2 = __riscv_vslide1up_vx_i8m8(u2, 0, vl);
      // vint8m8_t v2 = __riscv_vslide1up_vx_i8m8(v2, 0, vl);

      // // 1,1,2,2,3,3,4,4
      // vint8m8_t u =  __riscv_vadd_vx_i8m8(__riscv_vadd_vv_i8m8(u1, u2, vl), 1, vl);
      // vint8m8_t v =  __riscv_vadd_vx_i8m8(__riscv_vadd_vv_i8m8(v1, v2, vl), 1, vl);

      // // __riscv_vget_v_i8m8_i8m4 = vint8m4_t
      // vint16m8_t u_lo = __riscv_vsext_vf2_i16m8(__riscv_vget_v_i8m8_i8m4(u, 0), vl / 2);
      // vint16m8_t u_hi = __riscv_vsext_vf2_i16m8(__riscv_vget_v_i8m8_i8m4(u, vl / 2), vl / 2);
      w+= vl / 2;
    } while (w < image->width / 2);
      y0_ptr += image->luma_stride * 2;
      y1_ptr += image->luma_stride * 2;
      u_ptr += image->chroma_stride;
      v_ptr += image->chroma_stride;
  } while (++h < image->height / 2);
}

// void transformYuv420_neon(jr_uncompressed_ptr image, const int16_t* coeffs_ptr) {
//   // Implementation assumes image buffer is multiple of 16.
//   assert(image->width % 16 == 0);
//   uint8_t* y0_ptr = static_cast<uint8_t*>(image->data);
//   uint8_t* y1_ptr = y0_ptr + image->luma_stride;
//   uint8_t* u_ptr = static_cast<uint8_t*>(image->chroma_data);
//   uint8_t* v_ptr = u_ptr + image->chroma_stride * (image->height / 2);
//   size_t vl = 8;
//   // const vint16m8_t coeffs = __riscv_vle16_v_i16m8(coeffs_ptr, vl);

//   // 加载 8 个 16 位
//    const int16x8_t coeffs = vld1q_s16(coeffs_ptr);
//   // const vuint16m8_t uv_bias = __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vmv_v_v_i16m8(-128, vl), vl);

//   // int16 的 -128 转换成 8个i16 = v128
//   // i16 to u16
//    const uint16x8_t uv_bias = vreinterpretq_u16_s16(vdupq_n_s16(-128));
//   size_t h = 0;
//   do {
//     size_t w = 0;
//     do {
//       // vuint8m8_t y0 = __riscv_vle8_v_u8m8(y0_ptr + w * 2, vl);

//       // 16 个 8 位 uint  = v128
//       uint8x16_t y0 = vld1q_u8(y0_ptr + w * 2);
//       uint8x16_t y1 = vld1q_u8(y1_ptr + w * 2);

//       // 8 个 8 位 uint = v64
//       uint8x8_t u = vld1_u8(u_ptr + w);
//       uint8x8_t v = vld1_u8(v_ptr + w);

//       // 128 bias for UV given we are using libjpeg; see:
//       // https://github.com/kornelski/libjpeg/blob/master/structure.doc

//       // vaddw_u8: 8 位无符号整数相加, 应该会将 u8扩展成u16
//       int16x8_t u_wide_s16 = vreinterpretq_s16_u16(vaddw_u8(uv_bias, u));  // -128 + u
//       int16x8_t v_wide_s16 = vreinterpretq_s16_u16(vaddw_u8(uv_bias, v));  // -128 + v

//       // 好像是复制一遍，1，2，3，4 变成 1，1，2，2，3，3，4，4
//       const int16x8_t u_wide_lo = vzip1q_s16(u_wide_s16, u_wide_s16);
//       const int16x8_t u_wide_hi = vzip2q_s16(u_wide_s16, u_wide_s16);
//       const int16x8_t v_wide_lo = vzip1q_s16(v_wide_s16, v_wide_s16);
//       const int16x8_t v_wide_hi = vzip2q_s16(v_wide_s16, v_wide_s16);

//       const int16x8_t y0_lo = yConversion_neon(vget_low_u8(y0), u_wide_lo, v_wide_lo, coeffs);
//       const int16x8_t y0_hi = yConversion_neon(vget_high_u8(y0), u_wide_hi, v_wide_hi, coeffs);
//       const int16x8_t y1_lo = yConversion_neon(vget_low_u8(y1), u_wide_lo, v_wide_lo, coeffs);
//       const int16x8_t y1_hi = yConversion_neon(vget_high_u8(y1), u_wide_hi, v_wide_hi, coeffs);

//       const int16x8_t new_u = uConversion_neon(u_wide_s16, v_wide_s16, coeffs);
//       const int16x8_t new_v = vConversion_neon(u_wide_s16, v_wide_s16, coeffs);

//       // Narrow from 16-bit to 8-bit with saturation.
//       const uint8x16_t y0_output = vcombine_u8(vqmovun_s16(y0_lo), vqmovun_s16(y0_hi));
//       const uint8x16_t y1_output = vcombine_u8(vqmovun_s16(y1_lo), vqmovun_s16(y1_hi));
//       const uint8x8_t u_output = vqmovun_s16(vaddq_s16(new_u, vdupq_n_s16(128)));
//       const uint8x8_t v_output = vqmovun_s16(vaddq_s16(new_v, vdupq_n_s16(128)));

//       vst1q_u8(y0_ptr + w * 2, y0_output);
//       vst1q_u8(y1_ptr + w * 2, y1_output);
//       vst1_u8(u_ptr + w, u_output);
//       vst1_u8(v_ptr + w, v_output);

//       w += 8;
//     } while (w < image->width / 2);
//     y0_ptr += image->luma_stride * 2;
//     y1_ptr += image->luma_stride * 2;
//     u_ptr += image->chroma_stride;
//     v_ptr += image->chroma_stride;
//   } while (++h < image->height / 2);
// }

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

status_t convertYuv_rvv(jr_uncompressed_ptr image, ultrahdr_color_gamut src_encoding,
                          ultrahdr_color_gamut dst_encoding) {
   if (image == nullptr) {
     return ERROR_JPEGR_BAD_PTR;
   }
   if (src_encoding == ULTRAHDR_COLORGAMUT_UNSPECIFIED ||
       dst_encoding == ULTRAHDR_COLORGAMUT_UNSPECIFIED) {
     return ERROR_JPEGR_INVALID_COLORGAMUT;
   }

   const int16_t* coeffs = nullptr;
   switch (src_encoding) {
     case ULTRAHDR_COLORGAMUT_BT709:
       switch (dst_encoding) {
         case ULTRAHDR_COLORGAMUT_BT709:
           return JPEGR_NO_ERROR;
         case ULTRAHDR_COLORGAMUT_P3:
           coeffs = kYuv709To601_coeffs_rvv;
           break;
         case ULTRAHDR_COLORGAMUT_BT2100:
           coeffs = kYuv709To2100_coeffs_rvv;
           break;
         default:
           // Should be impossible to hit after input validation
           return ERROR_JPEGR_INVALID_COLORGAMUT;
       }
       break;
     case ULTRAHDR_COLORGAMUT_P3:
       switch (dst_encoding) {
         case ULTRAHDR_COLORGAMUT_BT709:
           coeffs = kYuv601To709_coeffs_rvv;
           break;
         case ULTRAHDR_COLORGAMUT_P3:
           return JPEGR_NO_ERROR;
         case ULTRAHDR_COLORGAMUT_BT2100:
           coeffs = kYuv601To2100_coeffs_rvv;
           break;
         default:
           // Should be impossible to hit after input validation
           return ERROR_JPEGR_INVALID_COLORGAMUT;
       }
       break;
     case ULTRAHDR_COLORGAMUT_BT2100:
       switch (dst_encoding) {
         case ULTRAHDR_COLORGAMUT_BT709:
           coeffs = kYuv2100To709_coeffs_rvv;
           break;
         case ULTRAHDR_COLORGAMUT_P3:
           coeffs = kYuv2100To601_coeffs_rvv;
           break;
         case ULTRAHDR_COLORGAMUT_BT2100:
           return JPEGR_NO_ERROR;
         default:
           // Should be impossible to hit after input validation
           return ERROR_JPEGR_INVALID_COLORGAMUT;
       }
       break;
     default:
       // Should be impossible to hit after input validation
       return ERROR_JPEGR_INVALID_COLORGAMUT;
   }

   if (coeffs == nullptr) {
     // Should be impossible to hit after input validation
     return ERROR_JPEGR_INVALID_COLORGAMUT;
   }

   transformYuv420_rvv(image, coeffs);
   return JPEGR_NO_ERROR;
 }
}  // namespace ultrahdr

