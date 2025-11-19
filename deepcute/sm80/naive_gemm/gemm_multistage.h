#include <cute/tensor.hpp>

using namespace cute;

template <class CTATile, int Stages>
struct GemmMultiStageSM80{
    // dtype
    using Dtype = half_t;

    // mma atom
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;

    // copy atom
    // g2s copy
    using g2s_copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Dtype>;
    // s2r copy
    using s2r_copy_atom = Copy_Atom<SM75_U32x4_LDSM_N, Dtype>;
    // r2s copy
    using r2s_copy_atom = Copy_Atom<UniversalCopy<uint32_t>, Dtype>;
    // s2g copy
    using s2g_copy_atom = Copy_Atom<UniversalCopy<uint128_t>, Dtype>;

    // we build everything based on (32, 32, 16) tile
    static_assert(size<0>(CTATile{}) % 32 == 0);
    static_assert(size<1>(CTATile{}) % 32 == 0);
    static_assert(size<2>(CTATile{}) % 16 == 0);

    // smem swizzle
    using SmemAtomAB = decltype(composition(Swizzle<2, 3, 3>{},
                                            make_layout(Shape<_32,_32>{}, 
                                                        Stride<_32,_1>{})));
    using SmemAtomC = decltype(composition(Swizzle<2, 3, 3>{},
                                            make_layout(Shape<_32,_32>{}, 
                                                        Stride<_32,_1>{})));
    using SmemLayoutA = decltype(tile_to_shape(SmemAtomAB{},
                                                make_shape(size<0>(CTATile{}), size<2>(CTATile{}), Int<Stages>{})));
    using SmemLayoutB = decltype(tile_to_shape(SmemAtomAB{},
                                                make_shape(size<1>(CTATile{}), size<2>(CTATile{}), Int<Stages>{})));
    using SmemLayoutC = decltype(tile_to_shape(SmemAtomC{},
                                                make_shape(_32{}, _32{}, _2{})));
    
    // tiled mma
    using TiledMMA = decltype(make_tiled_mma(mma_op{},
                                             make_layout(make_shape(_2{}, _2{})), // warp layout
                                             make_tile(_32{}, _32{}, _16{}))); // tile mnk

    // tiled copy
    // g2s tiled (32, 32)
    using G2STiledCopy = decltype(make_tiled_copy(g2s_copy_atom{},
                                                  make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})), // thr layout
                                                  make_layout(make_shape(_1{}, _8{})))); // val layout
    // s2r A & B (32, 16)
    using S2RTiledCopyA = decltype(make_tiled_copy_A(s2r_copy_atom{}, TiledMMA{}));
    using S2RTiledCopyB = decltype(make_tiled_copy_B(s2r_copy_atom{}, TiledMMA{}));
    // r2s C tiled (32, 32)
    using R2STiledCopy = decltype(make_tiled_copy_C(r2s_copy_atom{}, TiledMMA{}));
    // s2g tiled (32, 32)
    using S2GTiledCopy = decltype(make_tiled_copy(s2g_copy_atom{},
                                                  make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})), // thr layout
                                                  make_layout(make_shape(_1{}, _8{})))); // val layout

    CUTLASS_DEVICE
    void operator()(void* __restrict__ Aptr,
                    void* __restrict__ Bptr,
                    void* __restrict__ Cptr,
                    int M, int N, int K) {
        // smem allocation
        using T = half_t;            
        extern __shared__ T smem_data[];                        
        T *smem_A = smem_data;
        T *smem_B = smem_data + cosize(SmemLayoutA{});
        Tensor sA = make_tensor(make_smem_ptr<T>(smem_A), SmemLayoutA{}); // (cta_m, cta_k, stages)
        Tensor sB = make_tensor(make_smem_ptr<T>(smem_B), SmemLayoutB{}); // (cta_n, cta_k, stages)
        Tensor sC = make_tensor(make_smem_ptr<T>(smem_A), SmemLayoutC{}); // (cta_m, cta_n)

        Tensor A = make_tensor(make_gmem_ptr<T>(Aptr), make_layout(make_shape(M, K), make_stride(K, _1{})));
        Tensor B = make_tensor(make_gmem_ptr<T>(Bptr), make_layout(make_shape(N, K), make_stride(K, _1{})));
        Tensor C = make_tensor(make_gmem_ptr<T>(Cptr), make_layout(make_shape(M, N), make_stride(N, _1{})));

        // CTA tile (128, 128, 32)
        int idx = threadIdx.x;
        int ix = blockIdx.x;
        int iy = blockIdx.y;
        constexpr int cta_m = size<0>(CTATile{});
        constexpr int cta_n = size<1>(CTATile{});
        constexpr int cta_k = size<2>(CTATile{});
        Tensor gA = local_tile(A, make_tile(Int<cta_m>{}, Int<cta_k>{}), make_coord(iy, _)); // (cta_m, cta_k, rest_k)
        Tensor gB = local_tile(B, make_tile(Int<cta_n>{}, Int<cta_k>{}), make_coord(ix, _)); // (cta_n, cta_k, rest_k)
        Tensor gC = local_tile(C, make_tile(Int<cta_m>{}, Int<cta_n>{}), make_coord(iy, ix)); // (cta_m, cta_n)

        // register allocation
        TiledMMA tiled_mma; // (32, 16, 16)
        auto thr_mma = tiled_mma.get_slice(idx);
        auto t_rA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (8, 128/32, 32/16)
        auto t_rB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (4, 128/16, 32/16)
        auto t_rC = thr_mma.partition_fragment_C(gC(_, _)); // (8, 128/32, 128/16)
        clear(t_rC);

        // g2s copy partition
        G2STiledCopy tiled_g2s; // (32, 32)
        auto thr_g2s = tiled_g2s.get_slice(idx);
        auto t_g2s_gA = thr_g2s.partition_S(gA); // (8, 128/32, 32/32, num_k)
        auto t_g2s_sA = thr_g2s.partition_D(sA); // (8, 4, 1, stages)
        auto t_g2s_gB = thr_g2s.partition_S(gB); // (8, 4, 1, num_k)
        auto t_g2s_sB = thr_g2s.partition_D(sB); // (8, 4, 1, stages)

        // s2r copy partition
        S2RTiledCopyA tiled_s2r_A; // (32, 16)
        S2RTiledCopyB tiled_s2r_B; // (32, 16)
        auto thr_s2r_A = tiled_s2r_A.get_slice(idx);
        auto thr_s2r_B = tiled_s2r_B.get_slice(idx);
        auto t_s2r_sA = thr_s2r_A.partition_S(sA); // (8, 4, 2, stages)
        auto t_s2r_rA = thr_s2r_A.retile_D(t_rA); // (8, 4, 2)
        auto t_s2r_sB = thr_s2r_B.partition_S(sB); // (8, 4, 2, stages)
        auto t_s2r_rB = thr_s2r_B.retile_D(t_rB); // (8, 4, 2)

        // prepare before mainloop
        // 1. launch the stages - 1 copy
        // 2. launch s2r first small iter k copy
        constexpr int stages = Stages;
        int load_tile_idx = 0;
        int mma_tile_idx = 0;
        for (int istage=0; istage < stages - 1; istage++){
            copy(tiled_g2s, t_g2s_gA(_, _, _, istage), t_g2s_sA(_, _, _, istage));
            copy(tiled_g2s, t_g2s_gB(_, _, _, istage), t_g2s_sB(_, _, _, istage));
            cp_async_fence(); // commit
            load_tile_idx++;
        }
        cp_async_wait<stages - 2>();
        __syncthreads();
        copy(tiled_s2r_A, t_s2r_sA(_, _, 0, 0), t_s2r_rA(_, _, 0));
        copy(tiled_s2r_B, t_s2r_sB(_, _, 0, 0), t_s2r_rB(_, _, 0));

        // mainloop
        int num_k = size<3>(t_g2s_gA);
        int num_k_inner = size<2>(t_s2r_rA);
        int buffer_idx = 0;
        for (int itile = 0; itile < num_k; itile++) {
            // load next k tile
            if (load_tile_idx < num_k) {
                buffer_idx = load_tile_idx % stages;
                copy(tiled_g2s, t_g2s_gA(_, _, _, load_tile_idx), t_g2s_sA(_, _, _, buffer_idx));
                copy(tiled_g2s, t_g2s_gB(_, _, _, load_tile_idx), t_g2s_sB(_, _, _, buffer_idx));
                load_tile_idx++;
            }
            cp_async_fence();
            
            // small k iteration
            for (int ik = 0; ik < num_k_inner; ik++) {
                // load next small k tile
                if (ik == num_k_inner - 1){
                    // make sure the next k tile complete
                    cp_async_wait<stages - 2>();
                    __syncthreads();
                }
                int ik_next = (ik + 1) % num_k_inner;
                // calculate read tile
                int read_stage = (ik == num_k_inner - 1) ? (itile + 1) % stages : itile % stages;
                copy(tiled_s2r_A, t_s2r_sA(_, _, ik_next, read_stage), t_s2r_rA(_, _, ik_next));
                copy(tiled_s2r_B, t_s2r_sB(_, _, ik_next, read_stage), t_s2r_rB(_, _, ik_next));
                
                // gemm
                gemm(tiled_mma, t_rC, t_rA(_, _, ik), t_rB(_, _, ik), t_rC);
            }
        }

        // epilogue
        R2STiledCopy tiled_r2s; // (32, 32)
        auto thr_r2s = tiled_r2s.get_slice(idx);
        auto t_r2s_rC = thr_r2s.retile_S(t_rC); // (8, 4, 4)
        auto t_r2s_sC = thr_r2s.partition_D(sC); // (8, 1, 1, pipe)
        S2GTiledCopy tiled_s2g; // (32, 32)
        auto thr_s2g = tiled_s2g.get_slice(idx);
        auto t_s2g_sC = thr_s2g.partition_S(sC); // (8, 1, 1, pipe)
        auto t_s2g_gC = thr_s2g.partition_D(gC); // (8, 4, 4)
        // group modes
        auto t_r2s_rC_group = group_modes<1, 3>(t_r2s_rC); // (8, 16)
        auto t_s2g_gC_group = group_modes<1, 3>(t_s2g_gC); // (8, 16)
        // use 2 pipes for better batch request processing
        int pipe = size<3>(t_r2s_sC);
        for (int i = 0; i < size<1>(t_r2s_rC_group); i += pipe) {
            for (int j = 0; j < pipe; j++) {
                copy(tiled_r2s, t_r2s_rC_group(_, i + j), (t_r2s_sC(_, 0, 0, j)));
            }
            __syncthreads();
            for (int j = 0; j < pipe; j++) {
                copy(tiled_s2g, t_s2g_sC(_, 0, 0, j), t_s2g_gC_group(_, i + j));
            }
            __syncthreads();
        }
    }

    inline static cudaLaunchConfig_t get_launch_config(int M, int N, int K, cudaStream_t stream = 0) {
        constexpr int cta_m = size<0>(CTATile{});
        constexpr int cta_n = size<1>(CTATile{});
        constexpr int cta_k = size<2>(CTATile{});
        dim3 grid_dim(ceil_div(N, cta_n), ceil_div(M, cta_m), 1);
        dim3 block_dim(128, 1, 1); // fixed 128 threads
        size_t smem_size = (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{})) * sizeof(Dtype);
        return cudaLaunchConfig_t{grid_dim, block_dim, smem_size, stream};
    }
};