# Showcases. Where a demo shows one header, a showcase is one realistic program woven from many areas of core: a
# trading day, a shop's payments, a harbour crossing, an option priced four ways, a report written by the compiler.
# Each prints a coloured narrative, checks its results and exits non-zero on a mismatch, so each is a CTest
# (showcase.<name>, labels demo and showcase). They mirror no header, so scripts/check_mirror.py does not look here.
function(snicholls_core_add_showcase name)
    add_executable(snicholls_core_showcase_${name} ${name}.cpp)
    target_link_libraries(snicholls_core_showcase_${name} PRIVATE snicholls::core)
    snicholls_core_test_options(snicholls_core_showcase_${name})
    add_test(NAME showcase.${name} COMMAND snicholls_core_showcase_${name})
    set_tests_properties(showcase.${name} PROPERTIES LABELS "demo;showcase")
endfunction()

snicholls_core_add_showcase(functional_report)
snicholls_core_add_showcase(harbour_beacons)
snicholls_core_add_showcase(option_risk)
snicholls_core_add_showcase(payments_ledger)
snicholls_core_add_showcase(trading_day)
