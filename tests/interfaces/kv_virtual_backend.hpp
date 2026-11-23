//
//  kv_virtual_backend.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Test support: a backend in the virtual form that offers all three capabilities, as a real backend above core
//  would (one class deriving from each interface it implements), built over the reference store.
//

#ifndef snicholls_core_tests_interfaces_kv_virtual_backend_hpp
#define snicholls_core_tests_interfaces_kv_virtual_backend_hpp

#include <snicholls/core/interfaces/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_backup.hpp>
#include <snicholls/core/interfaces/kv_health.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace snicholls::kv_test {

    class kv_virtual_backend final : public key_value_store_interface, public kv_health_interface, public kv_backup_interface {
    public:
        reference_key_value_store store;

        kv_virtual_backend() = default;
        kv_virtual_backend(const kv_virtual_backend&) = delete;   // the adapters refer to this object's store
        kv_virtual_backend& operator=(const kv_virtual_backend&) = delete;

        result<bool, kv_errc> read(kv_bytes k, kv_read_visitor fn) const override { return kv_.read(k, fn); }
        result<bool, kv_errc> contains(kv_bytes k) const override { return kv_.contains(k); }
        result<std::size_t, kv_errc> scan(const kv_range& r, kv_scan_visitor fn) const override { return kv_.scan(r, fn); }
        result<std::uint64_t, kv_errc> sequence() const override { return kv_.sequence(); }
        result<void, kv_errc> put(kv_bytes k, kv_bytes v) override { return kv_.put(k, v); }
        result<bool, kv_errc> insert(kv_bytes k, kv_bytes v) override { return kv_.insert(k, v); }
        result<bool, kv_errc> erase(kv_bytes k) override { return kv_.erase(k); }
        result<std::size_t, kv_errc> erase_range(const kv_range& r) override { return kv_.erase_range(r); }
        result<void, kv_errc> apply(std::span<const kv_write> b) override { return kv_.apply(b); }
        result<std::unique_ptr<kv_reader_interface>, kv_errc> snapshot() const override { return kv_.snapshot(); }
        result<void, kv_errc> ping() const override { return health_.ping(); }
        result<kv_diagnostics, kv_errc> diagnostics() const override { return health_.diagnostics(); }
        result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view d) override { return backup_.checkpoint(d); }
        result<kv_verify_report, kv_errc> verify() const override { return backup_.verify(); }

    private:
        key_value_store_adapter<reference_key_value_store&> kv_{store};
        kv_health_adapter<reference_key_value_store&> health_{store};
        kv_backup_adapter<reference_key_value_store&> backup_{store};
    };
}

#endif /* snicholls_core_tests_interfaces_kv_virtual_backend_hpp */
