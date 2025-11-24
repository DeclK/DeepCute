#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <type_traits>

#include "scheduler.h"
#include "pipeline_states.h"

using namespace cute;

template <typename CTATile, bool MultiCast, int Stages>
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
    using SmemLayoutA = decltype(tile_to_shape(SmemABAtom, make_shape(size<0>(CTATile{}), size<2>(CTATile{}), Int<Stages>))); 
    using SmemLayoutB = decltype(tile_to_shape(SmemABAtom, make_shape(size<1>(CTATile{}), size<2>(CTATile{}), Int<Stages>))); 
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
        MBarrier full_barriers[Stages];
        MBarrier empty_barriers[Stages];
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

        // choose only 1 thread to prefetch tma & init mbarriers
        // TODO: what's the difference of using thread(0, 0)?
        int tid = threadIdx.x;
        int warp_id = __shfl_sync(0xffffffff, tid / 32, 0);
        if (warp_id == 0 && elect_one_sync()) {
            prefetch_tma_descriptor(&tma_a);
            prefetch_tma_descriptor(&tma_b);
            prefetch_tma_descriptor(&tma_c);
            for (int i = 0; i < Stages; i++) {
                smem.full_barriers[i].init(1);
                smem.empty_barriers[i].init(size(ClusterShape{}) * size(TiledMMA{}) / 32);  // size(TiledMMA{}) = 128 * 2, wg threads for mma
            }
        }
        cutlass::arch::fence_barrier_init();

        // sync here
        MultiCast ? cluster_sync() : __syncthreads();

        // producer & consumer, wg0 producer, wg1,2 consumer
        int warpgroup_id = warp_id / 4;
        if (warpgroup_id == 0) {
            // TODO: test if we do not use this reg trick
            cutlass::arch::warpgroup_reg_dealloc<40>();
            if (warp_id == 0 && elect_one_sync()){
                producer(smem, tma_a, tma_b, tma_c, M, N, K);
            }
        }
        else {
            cutlass::arch::warpgroup_reg_alloc<232>();
            consumer(smem, tma_a, tma_b, tma_c, M, N, K);
        }
    }

    CUTE_DEVICE
    void producer(SharedStorage &smem,
                  TmaDescriptor const &tma_a,
                  TmaDescriptor const &tma_b,
                  TmaDescriptor const &tma_c,
                  int M, int N, int K
                  ){
        // make gmem/tma tensor
        Tensor A = tma_a.get_tma_tensor(make_shape(m, k));
        Tensor B = tma_b.get_tma_tensor(make_shape(n, k)); 
        // local tile
        Tensor gA = local_tile(A, CTATile{}, make_coord(_, _, _), Step<_1, X, _1>{}); // (cta_m, cta_k, m, k)
        Tensor gB = local_tile(B, CTATile{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (cta_n, cta_k, n, k)
        // make smem tensor
        auto sA = make_tensor(make_smem_ptr<T>(smem.A.data()), SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr<T>(smem.B.data()), SmemLayoutB{});

        // get slice according to block in cluster
        int cluster_id = MultiCast ? block_id_in_cluster() : 0;
        auto thr_g2s_a = tma_a.get_slice(0);
        auto thr_g2s_b = tma_b.get_slice(cluster_id);
        uint16_t mcast_mask = (1 << size<0>(ClusterShape{})) - 1;

        // partition data
        Tensor t_g2s_gA = thr_g2s_a.partition_S(gA); // ((cta_m, cta_k), 1, 1, m, k)
        Tensor t_g2s_sA = thr_g2s_a.partition_D(sA(_, _, 0)); // ((cta_m, cta_k), 1, 1)
        Tensor t_g2s_gB = thr_g2s_b.partition_S(gB); // ((cta_n, cta_k), 1, 1, n, k)
        Tensor t_g2s_sB = thr_g2s_b.partition_D(sB(_, _, 0)); // ((cta_n, cta_k), 1, 1)

        // scheduler
        PersistantScheduler scheduler;
        auto tile_info = scheduler.get_tile_id();

        // pipeline stages init, TODO: init empty phase to 1 instead
        PipelineStates<Stages> pipe_states{1, 0};

        // calculate expecting bytes
        constexpr int SMEM_A_PerStage = size(remove<2>(SmemLayoutA{})) * sizeof(T);
        constexpr int SMEM_B_PerStage = size(remove<2>(SmemLayoutB{})) * sizeof(T);
        constexpr int Expect_Btyes_AB = SMEM_A_PerStage + SMEM_B_PerStage;

        // producer mainloop
        int num_k = size<3>(gA);
        while (tile_info.is_valid) {

            for (int i = 0; i < num_k; i++) {
                smem.empty_barriers[i].wait(pipe_states.phase);
                smem.full_barriers[i].arrive_and_expect_tx(Expect_Btyes_AB);
                copy(tma.with(reinterpret_cast<uint64_t*>(smem.empty_barriers[i])),
                     t_g2s_gA(_, _, _, tile_info.m_idx, i), t_g2s_sA);
                copy(tma.with(reinterpret_cast<uint64_t*>(smem.empty_barriers[i])),
                     t_g2s_gA(_, _, _, tile_info.m_idx, i), t_g2s_sA);
                pipe_states++;
            }
            scheduler.advance_next_tile();
            tile_info = scheduler.get_tile_id();    // TODO: ansower if is this a deep copy?
        }

        // make sure all consumers have use data
        for (int i = 0; i < Stages; i++) {
            smem.empty_barriers[i].wait(pipe_states);
            pipe_states++;
        }
    }

    CUTE_DEVICE
    void consumer(){}


    inline static cudaLaunchConfig_t get_launch_config(){}

};