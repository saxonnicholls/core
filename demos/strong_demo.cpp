//
//  strong_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/strong.hpp>: prices and quantities that cannot be confused. Exits
//  non-zero if a value is not the one documented.
//

#include <snicholls/core/strong.hpp>

#include <cstdio>

//! [strong]
using price = snicholls::strong<double, struct price_tag>;
using quantity = snicholls::strong<double, struct quantity_tag>;
using notional = snicholls::strong<double, struct notional_tag>;

// The one place a price meets a quantity: stated, not guessed.
constexpr notional operator*(price p, quantity q) { return notional(p.value * q.value); }
//! [strong]

int main()
{
    std::printf("snicholls-core strong demo\n");
    constexpr price bid(189.20), ask(189.30);
    constexpr price mid = (bid + ask) / 2.0;
    constexpr notional value = mid * quantity(100);
    std::printf("  mid %.2f, notional %.0f\n", mid.value, value.value);
    const bool ok = value.value == 18925.0;
    std::printf(ok ? "strong demo: ok\n" : "strong demo: FAILED\n");
    return ok ? 0 : 1;
}
