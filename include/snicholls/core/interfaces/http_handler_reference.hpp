//
//  http_handler_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::route_table`: the standard-library reference `http_handler`, a router from a method and an exact
/// path to handlers.
/// \ingroup core
///
/// **Overview.** A handler of handlers: `add` registers one per method and exact path; `handle` answers with it, 404
/// for an unknown path, 405 with an `Allow` header for a known path and another method, `HEAD` as `GET` without the
/// body, and 500 with the fixed body `internal error` when a handler throws, so nothing of the exception (its message
/// may carry user data or a secret) reaches the response. It is what a product's tests call routes through.
/// `add_pattern` (0.27.0, from crtp's `http_server_interface::route`) registers a pattern whose `:name` segments
/// capture into the request's `path_params` (`match_route`); exact paths win, then patterns in registration order.
///
/// \snippet interfaces/http_handler_reference_demo.cpp reference
///
/// **Relationships.** Models `http_handler` (`concepts/http_handler.hpp`); passes
/// `interfaces/http_handler_conformance.hpp` in every form. Group: \ref core.
///
/// **Implementations / variants.** Exact paths only; path parameters and middleware belong to the routers above core.
///
/// **Guarantees.** `handle` never throws except `std::bad_alloc`; it is `const` over the table, so one table may serve
/// many threads once built if its handlers are thread-safe. `add` is not thread-safe. Lookup is O(log routes).
///
/// **Errors.** `add` returns `capability_errc::conflict` for a route already registered and `invalid_argument` for a
/// path not starting with `/` or an empty handler.
///
/// **Performance.** `benchmarks/interfaces/http_handler_benchmark.cpp`.
///
/// **Tests and demos.** `tests/interfaces/http_handler_reference_test.cpp`,
/// `demos/interfaces/http_handler_reference_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<map>`, `<new>`, `<string>`, `<utility>`, `<vector>`, `capability_error.hpp`,
/// `interfaces/http_handler.hpp`.
///
/// **Known issues.** Exact paths only.
///
/// **Status.** New in 0.23.0; `add_pattern` in 0.27.0.

#ifndef snicholls_core_interfaces_http_handler_reference_hpp
#define snicholls_core_interfaces_http_handler_reference_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/interfaces/http_handler.hpp>

#include <cstddef>
#include <functional>
#include <map>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The reference router.
    /// \ingroup core
    class route_table {
    public:
        /// \brief One route's handler.
        using handler = std::function<http_response(const http_request&)>;

        /// \brief Registers `h` for `method` on the exact `path`.
        /// \param method The method.
        /// \param path   The path, from `/`.
        /// \param h      The handler.
        /// \return Success, `conflict` or `invalid_argument`.
        capability_result<void> add(http_method method, std::string path, handler h)
        {
            if (path.empty() || path.front() != '/')
                return fail(capability_errc::invalid_argument, {}, "a path starts with /");
            if (!h)
                return fail(capability_errc::invalid_argument, {}, "an empty handler");
            auto& methods = routes_[std::move(path)];
            if (!methods.emplace(method, std::move(h)).second)
                return fail(capability_errc::conflict, {}, "the route is already registered");
            return ok();
        }

        /// \brief Registers `h` for `method` on a pattern whose `:name` segments capture (`/orders/:id`); exact routes
        /// are tried first, then patterns in the order they were added, and the first that matches answers.
        /// \param method  The method.
        /// \param pattern The pattern, from `/`.
        /// \param h       The handler; its request carries the captures in `path_params`.
        /// \return Success, `conflict` or `invalid_argument`.
        capability_result<void> add_pattern(http_method method, std::string pattern, handler h)
        {
            if (pattern.empty() || pattern.front() != '/')
                return fail(capability_errc::invalid_argument, {}, "a path starts with /");
            if (!h)
                return fail(capability_errc::invalid_argument, {}, "an empty handler");
            for (auto& [p, methods] : patterns_)
                if (p == pattern) {
                    if (!methods.emplace(method, std::move(h)).second)
                        return fail(capability_errc::conflict, {}, "the route is already registered");
                    return ok();
                }
            patterns_.emplace_back(std::move(pattern), std::map<http_method, handler>{});
            patterns_.back().second.emplace(method, std::move(h));
            return ok();
        }

        /// \brief Answers a request.
        /// \param r The request.
        /// \return The route's response, or 404, 405 or 500.
        http_response handle(const http_request& r) const
        {
            const auto path = routes_.find(r.path);
            if (path != routes_.end())
                return dispatch(path->second, r);
            for (const auto& [pattern, methods] : patterns_) {
                http_request captured = r;
                if (match_route(pattern, r.path, captured.path_params))
                    return dispatch(methods, captured);
            }
            return respond(404, "not found");
        }

        /// \brief The number of (method, path or pattern) routes.
        /// \return The count.
        [[nodiscard]] std::size_t size() const noexcept
        {
            std::size_t n = 0;
            for (const auto& [p, methods] : routes_)
                n += methods.size();
            for (const auto& [p, methods] : patterns_)
                n += methods.size();
            return n;
        }

    private:
        static http_response dispatch(const std::map<http_method, handler>& methods, const http_request& r)
        {
            auto route = methods.find(r.method);
            const bool head_as_get = route == methods.end() && r.method == http_method::head;
            if (head_as_get)
                route = methods.find(http_method::get);
            if (route == methods.end()) {
                std::string allow;
                for (const auto& [m, h] : methods)
                    allow += (allow.empty() ? "" : ", ") + std::string(to_string(m));
                http_response out = respond(405, "method not allowed");
                out.headers.push_back({"Allow", std::move(allow)});
                return out;
            }
            http_response out;
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            try {
                out = route->second(r);
            } catch (const std::bad_alloc&) {
                throw;
            } catch (...) {
                return respond(500, "internal error");
            }
#else
            out = route->second(r);
#endif
            if (head_as_get)
                out.body.clear();
            return out;
        }

        std::map<std::string, std::map<http_method, handler>, std::less<>> routes_;
        std::vector<std::pair<std::string, std::map<http_method, handler>>> patterns_;
    };

    static_assert(http_handler<route_table>, "snicholls::route_table must model snicholls::http_handler");

} // namespace snicholls

#endif // snicholls_core_interfaces_http_handler_reference_hpp
