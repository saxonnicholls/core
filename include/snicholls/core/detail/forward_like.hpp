//
//  forward_like.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

// Internal: like_t<T, U> is U with the constness and value category of the forwarding type T
// (the type computation of C++23 std::forward_like). T is as deduced for a forwarding reference:
// X& for lvalues, X for rvalues.

#ifndef snicholls_core_detail_forward_like_hpp
#define snicholls_core_detail_forward_like_hpp

#include <type_traits>

namespace snicholls::detail {

template <typename T, typename U>
using like_t = std::conditional_t<
    std::is_lvalue_reference_v<T>,
    std::conditional_t<std::is_const_v<std::remove_reference_t<T>>, const U&, U&>,
    std::conditional_t<std::is_const_v<std::remove_reference_t<T>>, const U&&, U&&>>;

} // namespace snicholls::detail

#endif /* snicholls_core_detail_forward_like_hpp */
