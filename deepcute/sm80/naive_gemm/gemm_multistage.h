#include <cute/tensor.hpp>

using namespace cute;

template <class CTATile, int Stages>
struct GemmMultiStageSM80{
    // dtype
    using Dtype = half_t;

    // mma atom
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    // copy atom
    // g2s copy
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, Dtype>;

    // s2r copy
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, Dtype>;

    // r2s copy
    using r2s_copy_op = UniversalCopy<uint32_t>;
    using r2s_copy_traits = Copy_Traits<r2s_copy_op>;
    using r2s_copy_atom = Copy_Atom<r2s_copy_traits, Dtype>;

    // s2g copy
    using s2g_copy_op = UniversalCopy<uint128_t>;
    using s2g_copy_traits = Copy_Traits<s2g_copy_op>;
    using s2g_copy_atom = Copy_Atom<s2g_copy_traits, Dtype>;

    // basic building tv is (32, 8), changes the basic building mnk shape
    // (16, 8, 16) -> (16, 16, 16). We use 4 warps (2, 2) to multiply mn
    // (16, 16, 16) -> (32, 32, 16) is the basic tile
    // we build everything based on this (32, 32, 16) tile
    // check cta tile is multiple of (32, 32, 16)
    static_assert(size<0>(CTATile{}) % 32 == 0);
    static_assert(size<1>(CTATile{}) % 32 == 0);
    static_assert(size<2>(CTATile{}) % 16 == 0);

    // smem swizzle
    using SmemAtomAB = decltype(composition(Swizzle<1, 3, 3>,
                                                  make_layout(Shape<_32,_16>{}, 
                                                              Stride<_16,_1>{})));
    using SmemAtomC = decltype(composition(Swizzle<2, 3, 3>,
                                                  make_layout(Shape<_32,_32>{}, 
                                                              Stride<_32,_1>{})));
    // tiled mma
    using TiledMMA = decltype(make_tiled_mma(mma_atom{},
                                             make_layout(make_shape(_2{}, _2{})), // warp layout
                                             make_layout(make_shape(_32{}, _32{}, _16{})))); // tile mnk

    // tiled copy
    // g2s tiled copy AB both use (32, 16) or (32, 32)?
    using G2STiledCopy = decltype(make_tiled_copy(g2s_copy_atom{},
    ))
};