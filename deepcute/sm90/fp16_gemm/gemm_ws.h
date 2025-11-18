#include <cute/tensor.hpp>
#include <type_traits>

using namespace cute;

template <typename CTATile, bool MultiCast, int Stage>
struct GemmFp16SM90 {
    using Dtype = half_t;
    // cluster shape
    using ClusterShape = std::conditional_t<Multicast, decltype(make_shape(_2{}, _1{}, _1{})),  decltype(make_shape(_1{}, _1{}, _1{}))>

    // mma atom
    using ABMajor = GMMA::Major::K;
    using mma_op = GMMA::MMA_64x128x16_F16F16F16_SS<ABMajor, ABMajor>;

    // copy atom
    // g2s copy, using tma
    using g2s_copy_atom = std::conditional_t<MultiCast, SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST>;
    // r2s copy
    using r2s_copy_atom = Copy_Atom<SM90_U32x4_STSM_N, Dtype>;
    // s2g copy
    using s2g_copy_atom = SM90_TMA_STORE;

    // we build everything on top of (128, 128, 64), expand m-axis with 2 warp groups
    // expand k-mode 4 times
    static_assert(size<0>(CTATile{}) % 128 == 0);
    static_assert(size<1>(CTATile{}) % 128 == 0);
    static_assert(size<2>(CTATile{}) % 64 == 0);

    // smem swizzle
    using SmemABAtom = GMMA::Layout_K_SW128_Atom<Dtype>; // 64 fp16 % 128Bytes == 0
    using SmemCAtom = GMMA::Layout_K_INTER_Atom<Dtype>;  // 16 fp16 % 16Byptes == 0
    using SmemLayoutA = decltype(tile_to_shape(SmemABAtom, make_shape(size<0>(CTATile{}), size<2>(CTATile{}), Int<Stage>))); 
    using SmemLayoutB = decltype(tile_to_shape(SmemABAtom, make_shape(size<1>(CTATile{}), size<2>(CTATile{}), Int<Stage>))); 
    // TODO: test other tile shape
    using SmemLayoutC = decltype(tile_to_shape(SmemABAtom, make_shape(_32{}, _32{}, _2{})));

    // tiled mma, 2 warp group
    using TiledMMA = decltype(make_tiled_mma(mma_op{}, make_layout(make_shape(_2{}, _1{}, _1{}))));

    // tiled copy
    // g2s AB
    using G2STiledCopyA = decltype(make_tma_copy_A_sm90(g2s_copy_atom{},
                                                 make_tensor(static_cast<const Dtype*>(nullptr),
                                                             make_layout(make_shape(0, 0), make_stride(0, _1{})), // fake tensor
                                                 SmemLayoutA{},
                                                 CTATile{},
                                                 ClusterShape{})));
    using G2STiledCopyB = decltype(make_tma_copy_B_sm90(g2s_copy_atom{},
                                                 make_tensor(static_cast<const Dtype*>(nullptr),
                                                             make_layout(make_shape(0, 0), make_stride(0, _1{})), // fake tensor
                                                 SmemLayoutB{},
                                                 CTATile{},
                                                 ClusterShape{})));

    // r2s C
    using R2STiledCopy = decltype(make_tiled_copy_C(r2s_copy_atom{}, TiledMMA{}));

    // s2g tiled copy
    // TODO: define Epilogue tile
    // using S2GTiledCopy = decltype(make_tma_copy_C_sm90(s2g_copy_atom{},
    //                                              make_tensor(static_cast<const Dtype*>(nullptr),
    //                                                          make_layout(make_shape(0, 0), make_stride(0, _1{})), // fake tensor
    //                                              SmemLayoutC{},
    //                                              CTATile{},
    //                                              ClusterShape{})));


};