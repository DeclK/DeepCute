#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
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
    using g2s_copy_atom_a = SM90_TMA_LOAD;
    using g2s_copy_atom_b = std::conditional_t<MultiCast, SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST>;
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
    using SmemCAtom = GMMA::Layout_K_SW128_Atom<Dtype>;
    using SmemLayoutA = decltype(tile_to_shape(SmemABAtom, make_shape(size<0>(CTATile{}), size<2>(CTATile{}), Int<Stage>))); 
    using SmemLayoutB = decltype(tile_to_shape(SmemABAtom, make_shape(size<1>(CTATile{}), size<2>(CTATile{}), Int<Stage>))); 
    // TODO: test other tile shape
    using SmemLayoutC = decltype(tile_to_shape(SmemCAtom, make_shape(_64{}, _64{}, _2{})));

    // tiled mma, 2 warp group
    using TiledMMA = decltype(make_tiled_mma(mma_op{}, make_layout(make_shape(_2{}, _1{}, _1{}))));

    // tiled copy
    // g2s AB, just a placeholder here, acutal defined in `build_tma_descriptor`
    using G2STmaCopyA = TmaDescriptor;
    using G2STmaCopyB = TmaDescriptor;
    // r2s C
    using R2STiledCopy = decltype(make_tiled_copy_C(r2s_copy_atom{}, TiledMMA{}));
    // s2g tiled copy
    using S2GTmaCopyC = TmaDescriptor; 

    
    CUTE_HOST
    auto build_tma_descriptor(Dtype* Aptr, Dtype* Bptr, Dtype* Cptr,
                              int M, int N, int K){
        using T = Dtype;
        Tensor A = make_tensor(make_gmem_ptr<T>(Aptr), make_layout(make_shape(M, K), make_stride(K, _1{})));
        Tensor B = make_tensor(make_gmem_ptr<T>(Bptr), make_layout(make_shape(N, K), make_stride(K, _1{})));
        Tensor C = make_tensor(make_gmem_ptr<T>(Cptr), make_layout(make_shape(M, N), make_stride(N, _1{})));
        auto tma_a = make_tma_copy(g2s_copy_atom_a{}, A, remove<2>(SmemLayoutA{}), size<1>(ClusterShape{}));
        auto tma_b = make_tma_copy(g2s_copy_atom_b{}, B, remove<2>(SmemLayoutB{}), size<0>(ClusterShape{}));
        auto tma_c = make_tma_copy(s2g_copy_atom{}, C, SmemLayoutC{}, size<2>(ClusterShape{}));
        return make_tuple(tma_a, tma_b, tma_c);
    }

    struct SharedStorage {
        using MBarrier = cutlass::arch::ClusterTransactionBarrier;
        array<Dtype, cosize(SmemLayoutA{})> A;
        array<Dtype, cosize(SmemLayoutB{})> B;
        array<Dtype, cosize(SmemLayoutC{})> C;
        MBarrier full_barriers[Stage];
        MBarrier empty_barriers[Stage];
        // TODO: check the align bytes here
    }
                                                 

    CUTE_DEVICE
    void operator()(const __grid_constant__ TmaDescriptor tma_a,
                    const __grid_constant__ TmaDescriptor tma_b,
                    const __grid_constant__ TmaDescriptor tma_c,
                    int M, int N, int K) {
        using T = Dtype;
        using MBarrier = cutlass::arch::ClusterTransactionBarrier;

        // smem align to 1024 bytes for swizzle-128B, check if this is legal
        extern __shared__ __align__(1024) SharedStorage smem; 
        // make smem tensor
        auto smem_A = make_tensor(make_smem_ptr<T>(smem.A.data()), SmemLayoutA{});
        auto smem_B = make_tensor(make_smem_ptr<T>(smem.B.data()), SmemLayoutB{});

    }


    CUTE_DEVICE
    void producer(){}


    CUTE_DEVICE
    void comsumer(){}


    inline static cudaLaunchConfig_t get_launch_config(){}

};