#include <cute/tensor.hpp>
#define endl print("\n")
using namespace cute;


template <class RTensor, class ThrMMA, class ThrCopy, class Shape>
CUTE_HOST_DEVICE constexpr auto
retile_C_impl(RTensor&& r_tensor, ThrMMA&& thr_mma, ThrCopy&& thr_copy, Shape&& shape) {
    auto empty_tensor = make_tensor(static_cast<half_t*>(nullptr), make_layout(shape));
    auto mma2gmem = thr_mma.partition_C(empty_tensor).layout();
    auto copy2gmem = thr_copy.partition_D(empty_tensor).layout();
    auto gmem2mma = left_inverse(mma2gmem);
    auto copy2mma = gmem2mma.compose(copy2gmem);
    auto copy2rmem = r_tensor.compose(copy2mma);
    return copy2rmem;
}


int main() {
    using Dtype = half_t;
    using AccType = half_t;
    using mma_op = GMMA::MMA_64x128x16_F16F16F16_SS<GMMA::Major::K,  GMMA::Major::K>;

    using TiledMMA = decltype(make_tiled_mma(mma_op{}, make_layout(make_shape(_2{}, _1{}, _1{}))));

    // tiled copy
    using r2s_copy_atom = Copy_Atom<SM90_U32x4_STSM_N, Dtype>;
    using r2s_copy_atom_2 = Copy_Atom<UniversalCopy<uint32_t>, Dtype>;
    using empty_tenor = decltype(make_tensor(static_cast<Dtype *>(nullptr), make_layout(Shape<int, int>{}, Stride<int, _1>{})));
    using R2STiledCopy = decltype(make_tiled_copy_C(r2s_copy_atom{}, TiledMMA{}));
    auto tiled_copy = make_tiled_copy_C_atom(r2s_copy_atom_2{}, TiledMMA{});
    auto thr_copy = tiled_copy.get_slice(0);

    // see the retile result
    half_t *A;
    auto gC = make_tensor(A, make_layout(make_shape(Int<128>{}, Int<128>{}), LayoutRight{}));

    // tiled mma
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(0);
    auto t_rC = thr_mma.partition_fragment_C(gC);
    print(t_rC); endl;    
    auto t_r2s_rC = thr_copy.retile_S(t_rC);
    auto output = retile_C_impl(t_rC, thr_mma, thr_copy, make_shape(Int<128>{}, Int<128>{}));
    print(output); endl;
    auto t_r2s_sC = thr_copy.partition_D(gC);
    print(t_r2s_rC);endl;
    print(t_r2s_sC);endl;

}