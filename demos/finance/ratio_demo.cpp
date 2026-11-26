//
//  ratio_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/ratio.hpp>: a split ratio, a demerger's cost apportionment and a repo rate as exact
//  ratios, a year fraction multiplied by a rate, the exact order, the checked arithmetic and the parse errors. Exits
//  non-zero if a result is not the one documented.
//

#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/finance/ratio.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    std::printf("snicholls-core finance::ratio demo\n");
    using namespace std::chrono;
    using namespace snicholls::finance;

    //! [ratio]
    constexpr ratio split{3, 2};                                    // 3 new for 2 old
    constexpr ratio apportioned = *ratio::parse("13.24%");          // the demerged entity's share of the cost base
    static_assert(apportioned == ratio{331, 2500});
    const ratio repo_rate = *ratio::parse("4.35%");                 // a year
    const year_fraction days = act_365_fixed::year_fraction(sys_days{2026y / November / 2d}, sys_days{2026y / December / 2d});
    const auto accrual = multiply(repo_rate, days);                 // the interest per unit of cash lent, exactly
    //! [ratio]

    std::printf("  split %lld/%lld, apportioned %lld/%lld, repo %lld/%lld over %lld/%lld of a year = %lld/%lld\n",
                static_cast<long long>(split.numerator()), static_cast<long long>(split.denominator()),
                static_cast<long long>(apportioned.numerator()), static_cast<long long>(apportioned.denominator()),
                static_cast<long long>(repo_rate.numerator()), static_cast<long long>(repo_rate.denominator()),
                static_cast<long long>(days.numerator()), static_cast<long long>(days.denominator()),
                static_cast<long long>(accrual->numerator()), static_cast<long long>(accrual->denominator()));
    check(accrual && *accrual == ratio{87 * 30, 2000 * 365}, "4.35% for 30 days ACT/365F is 2610/730000, reduced");
    check(split > ratio{1, 1} && apportioned < ratio{1, 1}, "a split is above one, an apportionment at most one");
    check(*ratio::parse("-6/4") == -ratio{3, 2} && ratio{6, -4}.denominator() == 2, "normalised: lowest terms, positive denominator");
    check(*add(apportioned, *subtract(ratio{1, 1}, apportioned)) == ratio{1, 1}, "the two parts of a cost base add to one");
    check(*ratio{-2, 3}.inverse() == ratio{-3, 2} && !ratio{}.inverse(), "zero has no inverse");
    check(ratio::make(1, 0).unwrap_err() == ratio_errc::zero_denominator, "make refuses a zero denominator");
    check(ratio::parse("1.5/2").unwrap_err() == ratio_errc::malformed, "parse refuses what is not a ratio");
    check(!multiply(ratio{INT64_MAX, 1}, ratio{2, 1}), "arithmetic that leaves 64 bits is std::nullopt");
    check(ratio{INT64_MAX - 1, INT64_MAX} > ratio{INT64_MAX - 2, INT64_MAX - 1}, "the order is exact at the edge of 64 bits");
    check(ratio{1, 8}.to_double() == 0.125 && ratio{-1, 8}.is_negative() && ratio{}.is_zero(), "to_double, is_negative, is_zero");
    return failures == 0 ? 0 : 1;
}
