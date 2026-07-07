#pragma once

#include <cstddef>
#include <cstdint>

namespace kernel::utils {

inline auto get_workgroupgroup_count(uint32_t totalThreads,  //
                                     uint32_t workgroupSize  //
                                     ) noexcept -> uint32_t {
    return (totalThreads + workgroupSize - 1) / workgroupSize;
}

}  // namespace kernel::utils
