//
//  harsanyi_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/game/harsanyi.hpp>: the dividends of a book's P&L in two factors, and Shapley as equal shares of them. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/game/harsanyi.hpp>

#include <cstdio>
#include <span>

int main()
{
    std::printf("snicholls-core harsanyi demo\n");
    namespace game = snicholls::game;
    bool ok = true;

    //! [dividends]
    // P&L when spot moves (player 0), when volatility moves (player 1), and both: the 0.5 left over
    // when both move is the cross term, which neither earns alone.
    constexpr std::array<double, 4> pnl{0.0, 2.0, 1.0, 3.5};
    constexpr auto dividends = game::harsanyi_dividends<2>(game::from_table(pnl));
    static_assert(dividends[0b11] == 0.5);

    // Shapley shares each dividend equally among its coalition: the cross term half each.
    const auto shares = game::shapley_from_dividends(std::span<const double>(dividends));
    //! [dividends]

    std::printf("  dividends: spot %.2f, vol %.2f, together %.2f\n", dividends[1], dividends[2], dividends[3]);
    std::printf("  shapley:   spot %.2f, vol %.2f\n", shares.values[0], shares.values[1]);
    ok &= shares.values[0] == 2.25 && shares.values[1] == 1.25;

    std::printf(ok ? "harsanyi demo: ok\n" : "harsanyi demo: FAILED\n");
    return ok ? 0 : 1;
}
