#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

typedef int DataType;

kernel void group_max(const device DataType* input_array [[ buffer(0) ]],
                      device DataType* output_array [[ buffer(1) ]],
                      uint position [[thread_position_in_grid]],
                      uint group_pos [[threadgroup_position_in_grid]],
                      uint simd_group_index [[simdgroup_index_in_threadgroup]],
                      uint thread_index [[thread_index_in_simdgroup]])
{
    // 1Thread Groupに32のSIMD groupがあるのでその各々の計算結果を格納するメモリを確保
    // see: Metal Shading Language Specification 4.4 threadgroup Address Space
    threadgroup DataType simd_sum_array[32];

    // 各スレッドのinput値の合計を求める
    DataType simd_group_max = simd_sum(input_array[position]);
    // 合計を一時保存
    simd_sum_array[simd_group_index] = simd_group_max;
    // Thread Group内のすべてのスレッドの合計の計算＆一時保存を待つ
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 一時保存した値をすべてを足す。
    // 1つのSIMD Groupにで32スレッドあるので、１つのSIMD Groupのみで処理。
    if (simd_group_index == 0) {
        output_array[group_pos] = simd_sum(simd_sum_array[thread_index]);
    }
}
