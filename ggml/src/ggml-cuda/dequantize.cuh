#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static __device__ __forceinline__ void dequantize_q8_16(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_16 * x = (const block_q8_16 *) vx;

    const float d = __half2float(x[ib].d);

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;

    if (ib == 0 && iqs == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        // Print raw bytes of d to verify half vs float interpretation
        uint16_t d_raw = *((uint16_t*)&x[0].d);
        float d_as_float_raw = *(float*)&x[0].d; // reads 4 bytes as float
        printf("Q8_16B IN  block[0]: d_raw=0x%04x d_half=%f d_4byte=%f qs=%d %d %d %d\n",
            d_raw, __half2float(x[0].d), d_as_float_raw,
            (int)x[0].qs[0], (int)x[0].qs[1], (int)x[0].qs[2], (int)x[0].qs[3]);
        // Also print a few more blocks to verify stride
        printf("Q8_16B IN  block[1]: d_raw=0x%04x d_half=%f qs=%d %d %d %d\n",
            *((uint16_t*)&x[1].d), __half2float(x[1].d),
            (int)x[1].qs[0], (int)x[1].qs[1], (int)x[1].qs[2], (int)x[1].qs[3]);
        // Print dequantized output values for first 8 elements
        printf("Q8_16B OUT dequant[0..7]=%f %f %f %f %f %f %f %f\n",
            x[0].qs[0]*__half2float(x[0].d), x[0].qs[1]*__half2float(x[0].d),
            x[0].qs[2]*__half2float(x[0].d), x[0].qs[3]*__half2float(x[0].d),
            x[0].qs[4]*__half2float(x[0].d), x[0].qs[5]*__half2float(x[0].d),
            x[0].qs[6]*__half2float(x[0].d), x[0].qs[7]*__half2float(x[0].d));
        // Print all 16 dequantized values of block[0]
        float d0 = __half2float(x[0].d);
        printf("Q8_16B OUT dequant[8..15]=%f %f %f %f %f %f %f %f\n",
            x[0].qs[8]*d0, x[0].qs[9]*d0, x[0].qs[10]*d0, x[0].qs[11]*d0,
            x[0].qs[12]*d0, x[0].qs[13]*d0, x[0].qs[14]*d0, x[0].qs[15]*d0);
    }
}
