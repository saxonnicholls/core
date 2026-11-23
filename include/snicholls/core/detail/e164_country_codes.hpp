//
//  e164_country_codes.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Generated E.164 country calling codes for phone_number.hpp. Never edit by hand.
/// \ingroup core
///
/// Written by scripts/generate_e164_country_codes.py from scripts/data/e164/country_codes.txt, extracted
/// 2026-11-23 from the metadata of libphonenumber in the Python package phonenumbers 9.0.44. The CTest
/// generated.e164_country_codes_up_to_date fails if this file differs from what the generator writes. Not
/// public API.

#ifndef snicholls_core_detail_e164_country_codes_hpp
#define snicholls_core_detail_e164_country_codes_hpp

#include <cstdint>

namespace snicholls::detail::e164
{

/// The release of phonenumbers the table was extracted from.
inline constexpr char source[] = "phonenumbers 9.0.44";

/// The date of the extract.
inline constexpr char snapshot_date[] = "2026-11-23";

/// One country calling code: the code, and whether a leading 0 after it is the trunk prefix.
struct country_code_entry
{
    std::uint16_t code;   ///< 1 to 999
    bool trunk_zero;   ///< a national significant number under this code never starts with 0
};

/// Every assigned country calling code, ascending; the regions sharing it follow in a comment.
inline constexpr country_code_entry country_codes[] = {
    {1, false},   // US AG AI AS BB BM BS CA DM DO GD GU JM KN KY LC MP MS PR SX TC TT VC VG VI
    {7, false},   // RU KZ
    {20, true},   // EG
    {27, true},   // ZA
    {30, false},   // GR
    {31, true},   // NL
    {32, true},   // BE
    {33, true},   // FR
    {34, false},   // ES
    {36, false},   // HU
    {39, false},   // IT VA
    {40, true},   // RO
    {41, true},   // CH
    {43, true},   // AT
    {44, true},   // GB GG IM JE
    {45, false},   // DK
    {46, true},   // SE
    {47, false},   // NO SJ
    {48, false},   // PL
    {49, true},   // DE
    {51, true},   // PE
    {52, false},   // MX
    {53, true},   // CU
    {54, true},   // AR
    {55, true},   // BR
    {56, false},   // CL
    {57, true},   // CO
    {58, true},   // VE
    {60, true},   // MY
    {61, true},   // AU CC CX
    {62, true},   // ID
    {63, true},   // PH
    {64, true},   // NZ
    {65, false},   // SG
    {66, true},   // TH
    {81, true},   // JP
    {82, true},   // KR
    {84, true},   // VN
    {86, true},   // CN
    {90, true},   // TR
    {91, true},   // IN
    {92, true},   // PK
    {93, true},   // AF
    {94, true},   // LK
    {95, true},   // MM
    {98, true},   // IR
    {211, true},   // SS
    {212, true},   // MA EH
    {213, true},   // DZ
    {216, false},   // TN
    {218, true},   // LY
    {220, false},   // GM
    {221, false},   // SN
    {222, false},   // MR
    {223, false},   // ML
    {224, false},   // GN
    {225, false},   // CI
    {226, false},   // BF
    {227, false},   // NE
    {228, false},   // TG
    {229, false},   // BJ
    {230, false},   // MU
    {231, true},   // LR
    {232, true},   // SL
    {233, true},   // GH
    {234, true},   // NG
    {235, false},   // TD
    {236, false},   // CF
    {237, false},   // CM
    {238, false},   // CV
    {239, false},   // ST
    {240, false},   // GQ
    {241, false},   // GA
    {242, false},   // CG
    {243, true},   // CD
    {244, false},   // AO
    {245, false},   // GW
    {246, false},   // IO
    {247, false},   // AC
    {248, false},   // SC
    {249, true},   // SD
    {250, true},   // RW
    {251, true},   // ET
    {252, true},   // SO
    {253, false},   // DJ
    {254, true},   // KE
    {255, true},   // TZ
    {256, true},   // UG
    {257, false},   // BI
    {258, false},   // MZ
    {260, true},   // ZM
    {261, true},   // MG
    {262, true},   // RE YT
    {263, true},   // ZW
    {264, true},   // NA
    {265, true},   // MW
    {266, false},   // LS
    {267, false},   // BW
    {268, false},   // SZ
    {269, false},   // KM
    {290, false},   // SH TA
    {291, true},   // ER
    {297, false},   // AW
    {298, false},   // FO
    {299, false},   // GL
    {350, false},   // GI
    {351, false},   // PT
    {352, false},   // LU
    {353, true},   // IE
    {354, false},   // IS
    {355, true},   // AL
    {356, false},   // MT
    {357, false},   // CY
    {358, true},   // FI AX
    {359, true},   // BG
    {370, true},   // LT
    {371, false},   // LV
    {372, false},   // EE
    {373, true},   // MD
    {374, true},   // AM
    {375, true},   // BY
    {376, false},   // AD
    {377, true},   // MC
    {378, false},   // SM
    {380, true},   // UA
    {381, true},   // RS
    {382, true},   // ME
    {383, true},   // XK
    {385, true},   // HR
    {386, true},   // SI
    {387, true},   // BA
    {389, true},   // MK
    {420, false},   // CZ
    {421, true},   // SK
    {423, true},   // LI
    {500, false},   // FK
    {501, false},   // BZ
    {502, false},   // GT
    {503, false},   // SV
    {504, false},   // HN
    {505, false},   // NI
    {506, false},   // CR
    {507, false},   // PA
    {508, true},   // PM
    {509, false},   // HT
    {590, true},   // GP BL MF
    {591, true},   // BO
    {592, false},   // GY
    {593, true},   // EC
    {594, true},   // GF
    {595, true},   // PY
    {596, true},   // MQ
    {597, false},   // SR
    {598, true},   // UY
    {599, false},   // CW BQ
    {670, false},   // TL
    {672, false},   // NF
    {673, false},   // BN
    {674, false},   // NR
    {675, false},   // PG
    {676, false},   // TO
    {677, false},   // SB
    {678, false},   // VU
    {679, false},   // FJ
    {680, false},   // PW
    {681, false},   // WF
    {682, false},   // CK
    {683, false},   // NU
    {685, false},   // WS
    {686, true},   // KI
    {687, false},   // NC
    {688, false},   // TV
    {689, false},   // PF
    {690, false},   // TK
    {691, false},   // FM
    {692, false},   // MH
    {800, false},   // 001
    {808, false},   // 001
    {850, true},   // KP
    {852, false},   // HK
    {853, false},   // MO
    {855, true},   // KH
    {856, true},   // LA
    {870, false},   // 001
    {878, false},   // 001
    {880, true},   // BD
    {881, false},   // 001
    {882, false},   // 001
    {883, false},   // 001
    {886, true},   // TW
    {888, false},   // 001
    {960, false},   // MV
    {961, true},   // LB
    {962, true},   // JO
    {963, true},   // SY
    {964, true},   // IQ
    {965, false},   // KW
    {966, true},   // SA
    {967, true},   // YE
    {968, false},   // OM
    {970, true},   // PS
    {971, true},   // AE
    {972, true},   // IL
    {973, false},   // BH
    {974, false},   // QA
    {975, false},   // BT
    {976, true},   // MN
    {977, true},   // NP
    {979, false},   // 001
    {992, false},   // TJ
    {993, false},   // TM
    {994, true},   // AZ
    {995, true},   // GE
    {996, true},   // KG
    {998, false},   // UZ
};

} // namespace snicholls::detail::e164

#endif /* snicholls_core_detail_e164_country_codes_hpp */
