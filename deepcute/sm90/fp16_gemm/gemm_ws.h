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
    using ClusterShape = std::conditional_t<MultiCast, decltype(make_shape(_2{}, _1{}, _1{})),  decltype(make_shape(_1{}, _1{}, _1{}))>;
    // scheduler
    using Scheduler = PersistantScheduler<Shape<int, int, int>, CTATile, ClusterShape>;

    // mma atom
    using mma_op = GMMA::MMA_64x128x16_F16F16F16_SS<GMMA::Major::K,  GMMA::Major::K>;

    // smem swizzle
    using SmemABAtom = GMMA::Layout_K_SW128_Atom<Dtype>; // (8, 64) 64 fp16 % 128Bytes == 0
    using SmemCAtom = GMMA::Layout_K_SW32_Atom<Dtype>;
    using SmemLayoutA = decltype(tile_to_shape(SmemABAtom{}, make_shape(size<0>(CTATile{}), size<2>(CTATile{}), Int<Stages>{}))); 
    using SmemLayoutB = decltype(tile_to_shape(SmemABAtom{}, make_shape(size<1>(CTATile{}), size<2>(CTATile{}), Int<Stages>{}))); 
    // TODO: bigger pad, more pipe
    using SmemLayoutC = decltype(tile_to_shape(SmemCAtom{}, make_shape(_128{}, _16{}, _2{}), Step<_1, _2, _3>{}));
    
    // copy atom
    // g2s copy, using tma
    using g2s_copy_atom_a = SM90_TMA_LOAD;
    using g2s_copy_atom_b = std::conditional_t<MultiCast, SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST>;
    // r2s copy
    using r2s_copy_atom = Copy_Atom<SM90_U32x4_STSM_N, Dtype>;
    // s2g copy
    using s2g_copy_atom = SM90_TMA_STORE;

    // tiled mma, 2 warp group (128, 128, 16)
    using TiledMMA = decltype(make_tiled_mma(mma_op{}, make_layout(make_shape(_2{}, _1{}, _1{}))));

    // tiled copy
    // g2s AB (128, 64), just a placeholder here, acutal defined in `build_tma_descriptor`
    using empty_tenor = decltype(make_tensor(static_cast<Dtype const *>(nullptr), make_layout(Shape<int, int>{}, Stride<int, _1>{})));
    using G2STmaCopyA = decltype(make_tma_copy(g2s_copy_atom_a{}, empty_tenor{}, SmemLayoutA{}(_, _, 0), size<1>(ClusterShape{})));
    using G2STmaCopyB = decltype(make_tma_copy(g2s_copy_atom_b{}, empty_tenor{}, SmemLayoutB{}(_, _, 0), size<0>(ClusterShape{})));
    // r2s C (128, 16), TODO: use my layout tv & tilermn
    using R2STiledCopy = decltype(make_tiled_copy_C_atom(r2s_copy_atom{}, TiledMMA{}));
    // s2g tiled copy (128, 16)
    using S2GTmaCopyC = decltype(make_tma_copy(s2g_copy_atom{}, empty_tenor{}, SmemLayoutC{}(_, _, 0)));

    // cta tile must be divisible by all second level tile
    static_assert(size<0>(CTATile{}) % 128 == 0);
    static_assert(size<1>(CTATile{}) % 128 == 0);
    static_assert(size<2>(CTATile{}) % 64 == 0);

    CUTE_HOST
    static auto build_tma_descriptor(Dtype* Aptr, Dtype* Bptr, Dtype* Cptr,
                              int M, int N, int K){
        using T = Dtype;
        Tensor A = make_tensor(make_gmem_ptr<T>(Aptr), make_layout(make_shape(M, K), make_stride(K, _1{})));
        Tensor B = make_tensor(make_gmem_ptr<T>(Bptr), make_layout(make_shape(N, K), make_stride(K, _1{})));
        Tensor C = make_tensor(make_gmem_ptr<T>(Cptr), make_layout(make_shape(M, N), make_stride(N, _1{})));
        auto tma_a = make_tma_copy(g2s_copy_atom_a{}, A, SmemLayoutA{}(_, _, 0), size<1>(ClusterShape{}));
        auto tma_b = make_tma_copy(g2s_copy_atom_b{}, B, SmemLayoutB{}(_, _, 0), size<0>(ClusterShape{}));
        auto tma_c = make_tma_copy(s2g_copy_atom{}, C, SmemLayoutC{}(_, _, 0), size<2>(ClusterShape{}));
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
    };

    CUTE_DEVICE
    void operator()(G2STmaCopyA tma_a,
                    G2STmaCopyB tma_b,
                    S2GTmaCopyC tma_c,
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
            prefetch_tma_descriptor(tma_a.get_tma_descriptor());
            prefetch_tma_descriptor(tma_b.get_tma_descriptor());
            prefetch_tma_descriptor(tma_c.get_tma_descriptor());
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
                  G2STmaCopyA const &tma_a,
                  G2STmaCopyB const &tma_b,
                  S2GTmaCopyC const &tma_c,
                  int M, int N, int K
                  ){
        // make gmem/tma tensor
        Tensor A = tma_a.get_tma_tensor(make_shape(M, K));
        Tensor B = tma_b.get_tma_tensor(make_shape(N, K)); 
        // local tile
        Tensor gA = local_tile(A, CTATile{}, make_coord(_, _, _), Step<_1, X, _1>{}); // (cta_m, cta_k, m, k)
        Tensor gB = local_tile(B, CTATile{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (cta_n, cta_k, n, k)
        // make smem tensor
        auto sA = make_tensor(make_smem_ptr<Dtype>(smem.A.data()), SmemLayoutA{});
        auto sB = make_tensor(make_smem_ptr<Dtype>(smem.B.data()), SmemLayoutB{});

        // get slice according to block in cluster
        int cluster_id = MultiCast ? block_rank_in_cluster() : 0;
        auto thr_g2s_a = tma_a.get_slice(0);
        auto thr_g2s_b = tma_b.get_slice(cluster_id);
        uint16_t mcast_mask_b = (1 << size<0>(ClusterShape{})) - 1;

        // partition data
        Tensor t_g2s_gA = thr_g2s_a.partition_S(gA); // ((cta_m, cta_k), 1, 1, m, k)
        Tensor t_g2s_sA = thr_g2s_a.partition_D(sA(_, _, 0)); // ((cta_m, cta_k), 1, 1)
        Tensor t_g2s_gB = thr_g2s_b.partition_S(gB); // ((cta_n, cta_k), 1, 1, n, k)
        Tensor t_g2s_sB = thr_g2s_b.partition_D(sB(_, _, 0)); // ((cta_n, cta_k), 1, 1)

        // scheduler
        Scheduler scheduler(make_shape(M, N, K), 4);
        auto tile_info = scheduler.get_tile_id();

        // pipeline stages init, TODO: init empty phase to 1 instead
        PipelineStates<Stages> pipe_states{1, 0};

        // calculate expecting bytes
        constexpr int SMEM_A_PerStage = size(SmemLayoutA{}(_, _, 0)) * sizeof(Dtype);
        constexpr int SMEM_B_PerStage = size(SmemLayoutB{}(_, _, 0)) * sizeof(Dtype);
        constexpr int Expect_Btyes_AB = SMEM_A_PerStage + SMEM_B_PerStage;

        // producer mainloop
        int num_k = size<3>(gA);
        while (tile_info.is_valid) {

            for (int i = 0; i < num_k; i++) {
                smem.empty_barriers[pipe_states.stage_idx].wait(pipe_states.phase);
                smem.full_barriers[pipe_states.stage_idx].arrive_and_expect_tx(Expect_Btyes_AB);
                copy(tma_a.with(*reinterpret_cast<uint64_t*>(&smem.full_barriers[i]), 0),
                     t_g2s_gA(_, _, _, tile_info.m_idx, i), t_g2s_sA);
                copy(tma_b.with(*reinterpret_cast<uint64_t*>(&smem.full_barriers[i]), mcast_mask_b),
                     t_g2s_gB(_, _, _, tile_info.m_idx, i), t_g2s_sB);
                pipe_states++;
            }
            scheduler.advance_next_tile();
            tile_info = scheduler.get_tile_id();    // TODO: ansower if is this a deep copy?
        }

        // make sure all consumers have use data
        for (int i = 0; i < Stages; i++) {
            smem.empty_barriers[i].wait(pipe_states.phase);
            pipe_states++;
        }
    }

    CUTE_DEVICE
    void consumer(SharedStorage &smem,
                  G2STmaCopyA const &tma_a,
                  G2STmaCopyB const &tma_b,
                  S2GTmaCopyC const &tma_c,
                  int M, int N, int K) {
        // make the smem tensor
        Tensor sA = make_tensor(make_smem_ptr<Dtype>(smem.A.data()), SmemLayoutA{});    // (cta_m, cta_k, k)
        Tensor sB = make_tensor(make_smem_ptr<Dtype>(smem.B.data()), SmemLayoutB{});    // (cta_n, cta_k, k)
        Tensor sC = make_tensor(make_smem_ptr<Dtype>(smem.C.data()), SmemLayoutC{});
        // make C gmem/tma tensor
        Tensor C = tma_c.get_tma_tensor(make_shape(M, N));
        Tensor gC = local_tile(C, CTATile{}, make_coord(_, _, _), Step<_1, _1, X>{}); // (cta_m, cta_n, m, n)

        Scheduler scheduler(make_shape(M, N, K), 4);
        PipelineStates<Stages> pipe_states{0, 0};

        // mma 
        TiledMMA tiled_mma;
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        int warp_group_id_in_consumer = __shfl_sync(0xffffffff, threadIdx.x / 128 - 1, 0);
        // auto thr_mma = tiled_mma.get_slice(threadIdx.x - 128); // TODO: try this
        auto thr_mma = tiled_mma.get_slice(warp_group_id_in_consumer * 128);
        // mma fragments
        Tensor t_rA = thr_mma.partition_fragment_A(sA);   // (1, rest_m, rest_k, stages) matrix descriptor
        Tensor t_rB = thr_mma.partition_fragment_B(sB);   // (1, rest_n, rest_k, stages)
        // can't use thr_mma.partition_fragment_C, because we do not build a normal global Ctensor
        // TODO: try to build a fake gmem tensor
        Tensor t_rC = partition_fragment_C(tiled_mma, take<0, 2>(CTATile{})); // (CPY, rest_m, rest_n)

        // ctas to launch arrive, make sure cta in the same cluster complete mma together
        uint32_t lane_idx = threadIdx.x % 32;
        uint32_t predicate = MultiCast ? lane_idx == 0 : lane_idx < 2;

        // try to do the loop, and see what is missing
        int num_k = K / size<2>(CTATile{});
        auto tile_info = scheduler.get_tile_id(); 
        while (tile_info.is_valid) {
            // wait the full barrier
            clear(t_rC);
            for (int i = 0; i < num_k; i++) {
                // do mma
                smem.full_barriers[pipe_states.stage_idx].wait(pipe_states.phase);
                warpgroup_fence_operand(t_rC);
                warpgroup_arrive();
                gemm(tiled_mma, t_rA(_, _, _, pipe_states.stage_idx), t_rB(_, _, _, pipe_states.stage_idx), t_rC);
                warpgroup_fence_operand(t_rC);
                warpgroup_commit_batch();
                pipe_states++;
                // wait for mma complete, and update empty barrier
                warpgroup_wait<0>();
                smem.empty_barriers[pipe_states.stage_idx].arrive(lane_idx, predicate);
            }
            
            // to save register C to smem first then use tma to copy to gmem
            R2STiledCopy tiled_r2s;
            auto thr_r2s = tiled_r2s.get_slice(threadIdx.x - 128);
            Tensor t_r2s_rC = thr_r2s.retile_S(t_rC);     // ((CPY, restv), CTA_M / MMA_M = 1, CTA_N / MMA_N = 1)
            Tensor t_r2s_sC = thr_r2s.partition_D(sC);    // (CPY, rest_n = EPI_M / MMA_M, rest_n, PIPE) (8, 1, 1, 2)
            Tensor t_r2s_rC_flatten = t_r2s_rC(repeat<2>(_), _, _);
            // tma_c itself is both the data & tiled copy
            auto thr_s2g = tma_c.get_slice(0);
            Tensor t_s2g_sC = thr_s2g.partition_S(sC(_, _, 0)); // ((epil_m, epil_n), 1, 1, 2)
            Tensor t_s2g_gC = thr_s2g.partition_D(gC); // ((epil_m, epil_n), cta_m/epil_m, cta_n/epil_n, m, n) ((128, 16), 1, 8, m, n)
            // group for better look index
            Tensor t_r2s_rC_group = group_modes<1, 4>(t_r2s_rC_flatten); // (8, 8)
            Tensor t_r2s_sC_group = group_modes<1, 4>(t_r2s_sC); // (8, 2)
            Tensor t_s2g_sC_group = group_modes<1, 3>(t_s2g_sC); // ((128, 16), 2)
            Tensor t_s2g_gC_group = group_modes<1, 3>(t_s2g_gC); // ((128, 16), 8, m, n)

            int num_pads = size<2>(t_r2s_rC);
            int pipe = size<3>(t_r2s_sC);
            bool tma_predicate = (threadIdx.x - 128) == 0;
            
            for (int i = 0; i < num_pads; i += pipe) {
                for (int j = 0; j < pipe; j++) {
                    copy(tiled_r2s, t_r2s_rC_group(_, i + j), t_r2s_sC_group(_, j));
                }
                tma_store_fence();
                // tma copy
                if (tma_predicate) {
                    for (int j = 0; j < pipe; j++) {
                        copy(tma_c, t_s2g_sC_group(_, j), t_s2g_gC_group(_, i + j, tile_info.m_idx, tile_info.n_idx));
                        tma_store_arrive(); // TODO: can we use tma_desc_commit_group?
                    }
                }
                tma_store_wait<0>();
            }
            
            scheduler.advance_next_tile();
            tile_info = scheduler.get_tile_id();
        }
    }

    inline static cudaLaunchConfig_t get_launch_config(cudaStream_t stream = 0){
        cudaLaunchConfig_t launch_config;
        cudaLaunchAttribute launch_attr;
        launch_config.gridDim = Scheduler::get_grid_dim();
        launch_config.blockDim = {3*128}; // 3 warpgroups
        launch_config.dynamicSmemBytes = sizeof(SharedStorage);
        launch_config.stream = stream;

        // cluster
        launch_attr.id = cudaLaunchAttributeClusterDimension;
        launch_attr.val.clusterDim = {size<0>(ClusterShape{}),
                                    size<1>(ClusterShape{}),
                                    size<2>(ClusterShape{})};
        launch_config.numAttrs = 1;
        launch_config.attrs = &launch_attr;

        return launch_config;
    }

};