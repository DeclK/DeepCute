// a simple threadblock swizzle scheduler
#include <cute/tensor.hpp>
using namespace cute;

class PersistantScheduler {
    uint32_t tile_idx;
    uint32_t step_size;
    

    template <typename ClusterShape, typename ProblemMNK, typename CTATile>
    struct Param {
        Param() {}
    };
    
    struct TileInfo {
        int m_idx;
        int n_idx;
        bool is_valid;
    };
    
    CUTE_DEVICE
    PersistantScheduler() {}
    
    CUTE_DEVICE
    TileInfo get_tile_id() {
        TileInfo empty;
        return empty;
    }
    
    CUTE_DEVICE
    void advance_next_tile() {
        tile_idx += step_size;
    }
};