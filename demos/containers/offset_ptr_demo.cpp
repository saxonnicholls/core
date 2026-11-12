//
//  offset_ptr_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/offset_ptr.hpp>.
//

// Demo: snicholls::containers::offset_ptr - a linked list built inside one buffer still walks after the buffer is
// copied to another address, as a mapped file reopened elsewhere would be. Exits non-zero if a result is not as printed.

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <new>

#include <snicholls/core/containers/offset_ptr.hpp>

namespace {
int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}
} // namespace

int main()
{
    using snicholls::containers::offset_ptr;
    std::printf("offset_ptr<node>: %zu bytes\n", sizeof(offset_ptr<int>));
    //! [list]
    struct node
    {
        int value;
        offset_ptr<node> next;   // the distance to the next node, not its address
    };
    alignas(node) static std::byte region[3 * sizeof(node)];
    node* n = reinterpret_cast<node*>(region);
    ::new (&n[0]) node{1, nullptr};
    ::new (&n[1]) node{2, nullptr};
    ::new (&n[2]) node{3, nullptr};
    n[0].next = &n[2];
    n[2].next = &n[1];

    alignas(node) static std::byte moved[sizeof region];
    std::memcpy(moved, region, sizeof region);   // the region is now somewhere else
    int sum = 0, steps = 0;
    for (const node* p = std::launder(reinterpret_cast<node*>(moved)); p; p = p->next.get()) {
        sum = sum * 10 + p->value;
        ++steps;
    }
    //! [list]
    expect(sum == 132 && steps == 3, "the moved list walks 1 -> 3 -> 2");
    offset_ptr<node> copy = n[0].next;
    expect(copy.get() == &n[2] && copy.offset() != n[0].next.offset(), "a copy recomputes its offset");
    expect(!offset_ptr<node>{} && offset_ptr<node>{}.offset() == 1, "null is offset 1");
    return failures == 0 ? 0 : 1;
}
