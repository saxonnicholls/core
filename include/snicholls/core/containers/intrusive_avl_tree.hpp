//
//  intrusive_avl_tree.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief An intrusive, iterative AVL tree whose linkage and ordering are policies.
/// \ingroup core
///
/// | Form | Use |
/// |---|---|
/// | `snicholls::containers::avl_hook<T>` | the linkage a node embeds (three pointers and a height byte) |
/// | `snicholls::containers::member_avl_traits<T, &T::hook, &T::key, Compare>` | the usual traits: member hook, member key |
/// | `snicholls::containers::intrusive_avl_tree<T, Traits>` | the tree |

#ifndef snicholls_core_containers_intrusive_avl_tree_hpp
#define snicholls_core_containers_intrusive_avl_tree_hpp

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

namespace snicholls::containers {

/// \brief The linkage a node embeds to be a member of one intrusive AVL tree.
/// \ingroup core
///
/// A node may embed several hooks to sit in several trees at once. `height == 0` means "not linked".
/// \tparam T the node type.
template <class T>
struct avl_hook
{
    T* left = nullptr;          ///< Left child (smaller keys under the tree's ordering).
    T* right = nullptr;         ///< Right child (larger keys).
    T* parent = nullptr;        ///< Parent, or `nullptr` for the root.
    std::uint8_t height = 0;    ///< Height of the subtree rooted here (a leaf is 1); 0 when not linked.

    /// \brief Whether the node is currently linked into a tree.
    /// \return `true` when linked.
    [[nodiscard]] constexpr bool is_linked() const noexcept { return height != 0; }
};

/// \brief The traits an intrusive AVL tree needs: where the hook lives, the key and the ordering.
/// \ingroup core
/// \tparam Traits the traits type (static members only).
/// \tparam T the node type.
template <class Traits, class T>
concept avl_traits = requires(T& n, const T& c, const typename Traits::key_type& k) {
    typename Traits::key_type;
    { Traits::hook(n) } -> std::same_as<avl_hook<T>&>;
    { Traits::key(c) } -> std::convertible_to<typename Traits::key_type>;
    { Traits::less(k, k) } -> std::convertible_to<bool>;
};

/// \brief Traits for the common case: a member hook, a member key and a comparator.
/// \ingroup core
/// \tparam T the node type.
/// \tparam Hook pointer to the `avl_hook<T>` member.
/// \tparam Key pointer to the key data member.
/// \tparam Compare a stateless strict weak ordering on the key (e.g. `std::less<>` or `std::greater<>`).
template <class T, avl_hook<T> T::*Hook, auto Key, class Compare = std::less<>>
struct member_avl_traits
{
    /// \brief The key type (the member's type without cv-qualifiers).
    using key_type = std::remove_cvref_t<decltype(std::declval<const T&>().*Key)>;

    /// \brief The hook of a node.
    /// \param n the node.
    /// \return its hook.
    static constexpr avl_hook<T>& hook(T& n) noexcept { return n.*Hook; }

    /// \brief The key of a node.
    /// \param n the node.
    /// \return its key.
    static constexpr const key_type& key(const T& n) noexcept { return n.*Key; }

    /// \brief The ordering.
    /// \param a left key.
    /// \param b right key.
    /// \return `true` when `a` orders before `b`.
    static constexpr bool less(const key_type& a, const key_type& b) noexcept { return Compare{}(a, b); }
};

/// \brief An intrusive AVL tree: the caller owns the nodes, the tree only links them.
/// \ingroup core
///
/// \overview
/// A balanced binary search tree over nodes that embed an `avl_hook`. It allocates nothing and never copies a node:
/// insert links a node the caller owns, erase unlinks it. Linkage (`Traits::hook`) and ordering (`Traits::less`)
/// are policies, so one implementation serves trees that differ only in direction - for example the bid tree
/// (`std::greater<>`) and the ask tree (`std::less<>`) of a limit order book, and its buy-stop and sell-stop trees,
/// which the reference implementations each duplicated by hand. `front()` is
/// always the first node in the ordering, so with the right comparator it is the best price.
///
/// Use it when nodes already live in a pool or an arena and you need ordered access, O(log n) insert/erase and
/// stable node addresses. Prefer `std::map` when the tree should own its values, and a sorted vector for small,
/// rarely modified sets.
///
/// \relations
/// - Used by limit order books (price levels and stop levels).
/// - Nodes usually come from `snicholls::containers::object_pool`.
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation. The hook and ordering are the variation points.
///
/// \contracts
/// - **Thread safety:** none internal; one writer, or many readers with no writer.
/// - **Exception safety:** every operation is `noexcept` provided `Traits::less` does not throw.
/// - **Complexity:** insert, erase, find, lower_bound: O(log n); front/back: O(1) (cached); next/prev: O(1)
///   amortised; height <= 1.44 log2(n + 2).
/// - **Ownership and lifetime:** nodes must outlive their membership; destroying the tree does not touch nodes
///   (call `clear()` first if the hooks will be reused).
/// - **Invariants:** BST order under `Traits::less`, parent links consistent, stored heights exact, every balance
///   factor in [-1, 1]; `validate()` checks them all in O(n).
/// \pre A node is inserted into at most one tree per hook, and keys are not modified while linked.
///
/// \errors
/// None: `insert` of a duplicate key returns the existing node and `false`.
///
/// \examples
/// \snippet intrusive_avl_tree_demo.cpp tree
///
/// \perf
/// Measured inside limit order books.
/// In core, benchmarks/containers/intrusive_avl_tree_benchmark.cpp compares it like for like with `std::set`, after a fairness
/// check; not yet measured in Release.
///
/// \testing
/// - Tests: tests/containers/intrusive_avl_tree_test.cpp (model-checked against `std::set`, validated after
///   every operation, both orderings, multiple hooks).
/// - Demo: demos/containers/intrusive_avl_tree_demo.cpp.
/// - Benchmark: benchmarks/containers/intrusive_avl_tree_benchmark.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// - Generic: it needs nothing but the standard library and the node type's hook.
///
/// \status
/// Experimental.
///
/// \tparam T the node type.
/// \tparam Traits an `avl_traits` type.
template <class T, class Traits>
    requires avl_traits<Traits, T>
class intrusive_avl_tree
{
public:
    using value_type = T;                               ///< The node type.
    using key_type = typename Traits::key_type;         ///< The key type.
    using traits_type = Traits;                         ///< The traits.
    using size_type = std::size_t;                      ///< Sizes.

    /// \brief Bidirectional iterator in tree order (`end()` is a null node; `--end()` is `back()`).
    template <bool Const>
    class basic_iterator
    {
    public:
        using iterator_category = std::bidirectional_iterator_tag;   ///< Bidirectional.
        using value_type = T;                                         ///< The node type.
        using difference_type = std::ptrdiff_t;                       ///< Distances.
        using pointer = std::conditional_t<Const, const T*, T*>;      ///< Node pointer.
        using reference = std::conditional_t<Const, const T&, T&>;    ///< Node reference.

        /// \brief A singular iterator.
        basic_iterator() noexcept = default;
        /// \brief An iterator at `node` (or end when null) of `tree`.
        /// \param node the node.
        /// \param tree the tree.
        basic_iterator(pointer node, const intrusive_avl_tree* tree) noexcept : node_(node), tree_(tree) {}
        /// \brief Conversion from a mutable iterator.
        /// \param other the iterator.
        template <bool C = Const, class = std::enable_if_t<C>>
        basic_iterator(const basic_iterator<false>& other) noexcept : node_(other.get()), tree_(other.tree()) {}

        /// \brief The node.
        /// \return a reference to it.
        reference operator*() const noexcept { return *node_; }
        /// \brief The node.
        /// \return a pointer to it.
        pointer operator->() const noexcept { return node_; }
        /// \brief The node pointer (null at end).
        /// \return the pointer.
        pointer get() const noexcept { return node_; }
        /// \brief The tree.
        /// \return the tree.
        const intrusive_avl_tree* tree() const noexcept { return tree_; }

        /// \brief Advance.
        /// \return `*this`.
        basic_iterator& operator++() noexcept
        {
            node_ = intrusive_avl_tree::next(const_cast<T*>(node_));
            return *this;
        }
        /// \brief Advance (postfix).
        /// \return the previous position.
        basic_iterator operator++(int) noexcept
        {
            auto copy = *this;
            ++*this;
            return copy;
        }
        /// \brief Step back (from end: to the last node).
        /// \return `*this`.
        basic_iterator& operator--() noexcept
        {
            node_ = node_ ? intrusive_avl_tree::prev(const_cast<T*>(node_)) : tree_->last_;
            return *this;
        }
        /// \brief Step back (postfix).
        /// \return the previous position.
        basic_iterator operator--(int) noexcept
        {
            auto copy = *this;
            --*this;
            return copy;
        }
        /// \brief Equality (same node).
        /// \param a left.
        /// \param b right.
        /// \return `true` when equal.
        friend bool operator==(const basic_iterator& a, const basic_iterator& b) noexcept { return a.node_ == b.node_; }

    private:
        pointer node_ = nullptr;
        const intrusive_avl_tree* tree_ = nullptr;
    };

    using iterator = basic_iterator<false>;        ///< Mutable iterator.
    using const_iterator = basic_iterator<true>;   ///< Const iterator.

    /// \brief An empty tree.
    intrusive_avl_tree() noexcept = default;
    intrusive_avl_tree(const intrusive_avl_tree&) = delete;              ///< Not copyable (nodes are linked once).
    intrusive_avl_tree& operator=(const intrusive_avl_tree&) = delete;   ///< Not copyable.

    /// \brief Take over `other`'s nodes; `other` becomes empty.
    /// \param other the source.
    intrusive_avl_tree(intrusive_avl_tree&& other) noexcept
        : root_(std::exchange(other.root_, nullptr)), first_(std::exchange(other.first_, nullptr)),
          last_(std::exchange(other.last_, nullptr)), size_(std::exchange(other.size_, 0))
    {
    }

    /// \brief Take over `other`'s nodes (this tree's nodes are forgotten, not unlinked).
    /// \param other the source.
    /// \return `*this`.
    intrusive_avl_tree& operator=(intrusive_avl_tree&& other) noexcept
    {
        root_ = std::exchange(other.root_, nullptr);
        first_ = std::exchange(other.first_, nullptr);
        last_ = std::exchange(other.last_, nullptr);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    /// \brief Number of linked nodes.
    /// \return the size.
    [[nodiscard]] size_type size() const noexcept { return size_; }
    /// \brief Whether the tree is empty.
    /// \return `true` when empty.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    /// \brief The root (null when empty).
    /// \return the root.
    [[nodiscard]] T* root() const noexcept { return root_; }
    /// \brief The first node in the ordering (null when empty), O(1).
    /// \return the first node.
    [[nodiscard]] T* front() const noexcept { return first_; }
    /// \brief The last node in the ordering (null when empty), O(1).
    /// \return the last node.
    [[nodiscard]] T* back() const noexcept { return last_; }
    /// \brief Height of the tree (0 when empty).
    /// \return the height.
    [[nodiscard]] int height() const noexcept { return h(root_); }

    /// \name Iteration
    ///@{
    /// \brief Iterator to the first node.
    /// \return the iterator.
    iterator begin() noexcept { return {first_, this}; }
    /// \brief Iterator past the last node.
    /// \return the iterator.
    iterator end() noexcept { return {nullptr, this}; }
    /// \brief Const iterator to the first node.
    /// \return the iterator.
    const_iterator begin() const noexcept { return {first_, this}; }
    /// \brief Const iterator past the last node.
    /// \return the iterator.
    const_iterator end() const noexcept { return {nullptr, this}; }
    ///@}

    /// \brief The node after `n` in the ordering (null after the last).
    /// \param n a linked node.
    /// \return the successor.
    static T* next(T* n) noexcept
    {
        if (T* r = hk(n).right) {
            while (T* l = hk(r).left) r = l;
            return r;
        }
        T* p = hk(n).parent;
        while (p && hk(p).right == n) {
            n = p;
            p = hk(p).parent;
        }
        return p;
    }

    /// \brief The node before `n` in the ordering (null before the first).
    /// \param n a linked node.
    /// \return the predecessor.
    static T* prev(T* n) noexcept
    {
        if (T* l = hk(n).left) {
            while (T* r = hk(l).right) l = r;
            return l;
        }
        T* p = hk(n).parent;
        while (p && hk(p).left == n) {
            n = p;
            p = hk(p).parent;
        }
        return p;
    }

    /// \brief The node with key equal to `k`, or null.
    /// \param k the key.
    /// \return the node or null. O(log n).
    [[nodiscard]] T* find(const key_type& k) const noexcept
    {
        T* n = root_;
        while (n) {
            const auto& nk = Traits::key(*n);
            if (Traits::less(k, nk)) n = hk(n).left;
            else if (Traits::less(nk, k)) n = hk(n).right;
            else return n;
        }
        return nullptr;
    }

    /// \brief The first node whose key does not order before `k`, or null.
    /// \param k the key.
    /// \return the node or null. O(log n).
    [[nodiscard]] T* lower_bound(const key_type& k) const noexcept
    {
        T* n = root_;
        T* result = nullptr;
        while (n) {
            if (Traits::less(Traits::key(*n), k)) n = hk(n).right;
            else {
                result = n;
                n = hk(n).left;
            }
        }
        return result;
    }

    /// \brief The first node whose key orders after `k`, or null.
    /// \param k the key.
    /// \return the node or null. O(log n).
    [[nodiscard]] T* upper_bound(const key_type& k) const noexcept
    {
        T* n = root_;
        T* result = nullptr;
        while (n) {
            if (Traits::less(k, Traits::key(*n))) {
                result = n;
                n = hk(n).left;
            } else n = hk(n).right;
        }
        return result;
    }

    /// \brief Link `node`; if a node with an equal key is already linked, link nothing.
    /// \param node an unlinked node.
    /// \return the linked node with that key and whether `node` was inserted. O(log n).
    /// \pre `node` is not linked through this hook, unless it is the node already holding its key (then nothing
    /// changes and `{node, false}` is returned).
    std::pair<T*, bool> insert(T& node) noexcept
    {
        const auto& k = Traits::key(node);
        T* p = root_;
        bool go_left = false;
        while (p) {
            const auto& pk = Traits::key(*p);
            if (Traits::less(k, pk)) {
                go_left = true;
                if (!hk(p).left) break;
                p = hk(p).left;
            } else if (Traits::less(pk, k)) {
                go_left = false;
                if (!hk(p).right) break;
                p = hk(p).right;
            } else {
                return {p, false};   // equal key: nothing touched (not even node's hook)
            }
        }
        auto& nh = hk(&node);
        nh.left = nh.right = nullptr;
        nh.parent = p;
        nh.height = 1;
        ++size_;
        if (!p) {
            root_ = first_ = last_ = &node;
            return {&node, true};
        }
        (go_left ? hk(p).left : hk(p).right) = &node;
        if (go_left && p == first_) first_ = &node;
        if (!go_left && p == last_) last_ = &node;
        retrace_insert(p);
        return {&node, true};
    }

    /// \brief Unlink `node`.
    /// \param node a node linked into this tree.
    /// O(log n).
    void erase(T& node) noexcept
    {
        T* n = &node;
        if (n == first_) first_ = next(n);
        if (n == last_) last_ = prev(n);
        auto& nh = hk(n);
        T* retrace_from;
        if (nh.left && nh.right) {
            // Replace n by its successor s (leftmost of the right subtree), structurally.
            T* s = nh.right;
            while (hk(s).left) s = hk(s).left;
            auto& sh = hk(s);
            T* s_right = sh.right;
            if (sh.parent == n) {
                retrace_from = s;
            } else {
                retrace_from = sh.parent;
                hk(sh.parent).left = s_right;
                if (s_right) hk(s_right).parent = sh.parent;
                sh.right = nh.right;
                hk(nh.right).parent = s;
            }
            sh.left = nh.left;
            hk(nh.left).parent = s;
            sh.parent = nh.parent;
            sh.height = nh.height;
            replace_child(nh.parent, n, s);
        } else {
            T* child = nh.left ? nh.left : nh.right;
            if (child) hk(child).parent = nh.parent;
            replace_child(nh.parent, n, child);
            retrace_from = nh.parent;
        }
        nh.left = nh.right = nh.parent = nullptr;
        nh.height = 0;
        --size_;
        retrace_erase(retrace_from);
    }

    /// \brief Unlink every node (O(n): hooks are reset so nodes can be reused).
    void clear() noexcept
    {
        clear_and_dispose([](T&) noexcept {});
    }

    /// \brief Unlink every node, then call `dispose(node)` on each (post-order, O(n)).
    /// \param dispose called once per node after it is unlinked (e.g. return it to a pool).
    template <class Dispose>
    void clear_and_dispose(Dispose&& dispose) noexcept
    {
        T* n = root_;
        while (n) {
            auto& nh = hk(n);
            if (nh.left) n = nh.left;
            else if (nh.right) n = nh.right;
            else {
                T* p = nh.parent;
                if (p) (hk(p).left == n ? hk(p).left : hk(p).right) = nullptr;
                nh.parent = nullptr;
                nh.height = 0;
                dispose(*n);
                n = p;
            }
        }
        root_ = first_ = last_ = nullptr;
        size_ = 0;
    }

    /// \brief Check every invariant (order, parent links, heights, balance, size, cached ends). O(n).
    /// \return `true` when the tree is a valid AVL tree.
    [[nodiscard]] bool validate() const noexcept
    {
        if (!root_) return size_ == 0 && !first_ && !last_;
        if (hk(root_).parent) return false;
        size_type count = 0;
        if (check(root_, count) < 0) return false;
        if (count != size_) return false;
        T* lo = root_;
        while (hk(lo).left) lo = hk(lo).left;
        T* hi = root_;
        while (hk(hi).right) hi = hk(hi).right;
        if (lo != first_ || hi != last_) return false;
        // In-order traversal is strictly increasing.
        for (T* a = first_; a; ) {
            T* b = next(a);
            if (b && !Traits::less(Traits::key(*a), Traits::key(*b))) return false;
            a = b;
        }
        return true;
    }

private:
    static avl_hook<T>& hk(T* n) noexcept { return Traits::hook(*n); }
    static const avl_hook<T>& hk(const T* n) noexcept { return Traits::hook(const_cast<T&>(*n)); }
    static int h(const T* n) noexcept { return n ? hk(n).height : 0; }

    static void update_height(T* n) noexcept
    {
        const int l = h(hk(n).left), r = h(hk(n).right);
        hk(n).height = static_cast<std::uint8_t>(1 + (l > r ? l : r));
    }

    void replace_child(T* parent, T* old_child, T* new_child) noexcept
    {
        if (!parent) root_ = new_child;
        else if (hk(parent).left == old_child) hk(parent).left = new_child;
        else hk(parent).right = new_child;
    }

    // Rotations return the new subtree root; heights of both moved nodes are recomputed.
    T* rotate_left(T* x) noexcept
    {
        T* y = hk(x).right;
        T* b = hk(y).left;
        hk(x).right = b;
        if (b) hk(b).parent = x;
        hk(y).parent = hk(x).parent;
        replace_child(hk(x).parent, x, y);
        hk(y).left = x;
        hk(x).parent = y;
        update_height(x);
        update_height(y);
        return y;
    }

    T* rotate_right(T* x) noexcept
    {
        T* y = hk(x).left;
        T* b = hk(y).right;
        hk(x).left = b;
        if (b) hk(b).parent = x;
        hk(y).parent = hk(x).parent;
        replace_child(hk(x).parent, x, y);
        hk(y).right = x;
        hk(x).parent = y;
        update_height(x);
        update_height(y);
        return y;
    }

    // Restore balance at n (whose children are balanced); returns the subtree's new root.
    T* rebalance(T* n) noexcept
    {
        const int bf = h(hk(n).left) - h(hk(n).right);
        if (bf > 1) {
            T* l = hk(n).left;
            if (h(hk(l).left) < h(hk(l).right)) rotate_left(l);
            return rotate_right(n);
        }
        if (bf < -1) {
            T* r = hk(n).right;
            if (h(hk(r).right) < h(hk(r).left)) rotate_right(r);
            return rotate_left(n);
        }
        update_height(n);
        return n;
    }

    void retrace_insert(T* n) noexcept
    {
        while (n) {
            const int old = hk(n).height;
            T* top = rebalance(n);
            if (hk(top).height == old) return;   // subtree height unchanged: ancestors unaffected
            n = hk(top).parent;
        }
    }

    void retrace_erase(T* n) noexcept
    {
        while (n) {
            const int old = hk(n).height;
            T* top = rebalance(n);
            if (hk(top).height == old) return;   // subtree height unchanged: ancestors unaffected
            n = hk(top).parent;
        }
    }

    // Returns the subtree height, or -1 on any violation.
    int check(const T* n, size_type& count) const noexcept
    {
        if (!n) return 0;
        ++count;
        const auto& nh = hk(n);
        if (nh.left && (hk(nh.left).parent != n || !Traits::less(Traits::key(*nh.left), Traits::key(*n)))) return -1;
        if (nh.right && (hk(nh.right).parent != n || !Traits::less(Traits::key(*n), Traits::key(*nh.right)))) return -1;
        const int l = check(nh.left, count);
        if (l < 0) return -1;
        const int r = check(nh.right, count);
        if (r < 0) return -1;
        if (l - r > 1 || r - l > 1) return -1;
        const int height = 1 + (l > r ? l : r);
        return height == nh.height ? height : -1;
    }

    T* root_ = nullptr;
    T* first_ = nullptr;
    T* last_ = nullptr;
    size_type size_ = 0;
};

} // namespace snicholls::containers

#endif /* snicholls_core_containers_intrusive_avl_tree_hpp */
