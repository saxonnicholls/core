/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "snicholls-core", "index.html", [
    [ "Why", "index.html#autotoc_md1", null ],
    [ "What is here", "index.html#autotoc_md2", [
      [ "Foundations", "index.html#autotoc_md3", null ],
      [ "Money, time and markets", "index.html#autotoc_md4", null ],
      [ "State machines and functional programming", "index.html#autotoc_md5", null ],
      [ "Mathematics and numerics", "index.html#autotoc_md6", null ],
      [ "Tracking, geometry and distances", "index.html#autotoc_md7", null ],
      [ "Capability contracts", "index.html#autotoc_md8", null ],
      [ "Machine learning (experimental)", "index.html#autotoc_md9", null ],
      [ "Tokenizer throughput", "index.html#autotoc_md10", null ]
    ] ],
    [ "Using it", "index.html#autotoc_md11", [
      [ "Two copies in one build", "index.html#autotoc_md12", null ]
    ] ],
    [ "Building and testing", "index.html#autotoc_md13", null ],
    [ "Licence", "index.html#autotoc_md14", null ],
    [ "snicholls::result - Rust's <span class=\"tt\">Result&lt;V, E&gt;</span>, zero overhead, rethrowable, with traced errors", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html", [
      [ "Decisions", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md16", null ],
      [ "Per-call choices", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md17", null ],
      [ "API at a glance", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md18", null ],
      [ "Layout guarantees (tests/result_test.cpp)", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md19", null ],
      [ "Performance", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md20", null ],
      [ "Interop", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md21", null ],
      [ "Open items", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md22", null ]
    ] ],
    [ "Patterns", "patterns.html", [
      [ "Principles behind every pattern", "patterns.html#autotoc_md23", null ],
      [ "Errors", "patterns.html#autotoc_md25", [
        [ "Expected failure is a value: <span class=\"tt\">result&lt;V, E&gt;</span>", "patterns.html#autotoc_md26", null ],
        [ "Errors for logs: <span class=\"tt\">traced&lt;E&gt;</span>", "patterns.html#autotoc_md27", null ]
      ] ],
      [ "Capabilities: a contract, its twin, a reference and a kit", "patterns.html#autotoc_md29", null ],
      [ "State machines", "patterns.html#autotoc_md31", [
        [ "A machine is a table", "patterns.html#autotoc_md32", null ],
        [ "Save, load and draw it", "patterns.html#autotoc_md33", null ],
        [ "Lifecycles that read as the domain", "patterns.html#autotoc_md34", null ],
        [ "Machines, functionally", "patterns.html#autotoc_md35", null ]
      ] ],
      [ "Presentation is an attribute of the type", "patterns.html#autotoc_md37", null ],
      [ "Printing", "patterns.html#autotoc_md39", [
        [ "One message, one write", "patterns.html#autotoc_md40", null ],
        [ "A type that prints itself", "patterns.html#autotoc_md41", null ]
      ] ],
      [ "Dispatch over types and values", "patterns.html#autotoc_md43", [
        [ "Overload sets, never <span class=\"tt\">if constexpr</span> chains", "patterns.html#autotoc_md44", null ],
        [ "Values with <span class=\"tt\">constexpr_for</span>, types with <span class=\"tt\">type_list</span>", "patterns.html#autotoc_md45", null ],
        [ "Arguments that exist only at run time", "patterns.html#autotoc_md46", null ]
      ] ],
      [ "Reflection over plain aggregates", "patterns.html#autotoc_md48", null ],
      [ "Tables built by the compiler", "patterns.html#autotoc_md50", null ],
      [ "Exact numbers, time and identity", "patterns.html#autotoc_md52", [
        [ "Money is exact", "patterns.html#autotoc_md53", null ],
        [ "Time is explicit", "patterns.html#autotoc_md54", null ],
        [ "Identity is a type", "patterns.html#autotoc_md55", null ],
        [ "Secrets do not leak", "patterns.html#autotoc_md56", null ]
      ] ],
      [ "Dated money: one cash flow", "patterns.html#autotoc_md58", null ],
      [ "Statistics that merge", "patterns.html#autotoc_md60", [
        [ "Randomness that reproduces", "patterns.html#autotoc_md61", null ]
      ] ],
      [ "Attribution: who moved the number", "patterns.html#autotoc_md63", null ],
      [ "Functional programming, at compile time", "patterns.html#autotoc_md65", [
        [ "Functions that return functions", "patterns.html#autotoc_md66", null ],
        [ "Folds, unfolds and laziness", "patterns.html#autotoc_md67", null ],
        [ "Type classes and effects", "patterns.html#autotoc_md68", null ],
        [ "OCaml's pipes and functors", "patterns.html#autotoc_md69", null ],
        [ "Parsers from combinators", "patterns.html#autotoc_md70", null ],
        [ "Classics, written in C++", "patterns.html#autotoc_md71", null ],
        [ "An interpreter: algebraic data types and pattern matching", "patterns.html#autotoc_md72", null ],
        [ "Effects: railway, state, reader and writer", "patterns.html#autotoc_md73", null ],
        [ "Rust's vocabulary", "patterns.html#autotoc_md74", null ],
        [ "Rewriting and point-wise functions", "patterns.html#autotoc_md75", null ]
      ] ],
      [ "Signals at compile time", "patterns.html#autotoc_md77", null ],
      [ "Fixed sizes for the device", "patterns.html#autotoc_md79", null ],
      [ "Containers keyed by identity", "patterns.html#autotoc_md81", null ],
      [ "Hashing: the right hash for the job", "patterns.html#autotoc_md83", null ],
      [ "Linear algebra: one interface, a backend chosen by type", "patterns.html#autotoc_md85", null ],
      [ "Machine learning as small, composable kernels", "patterns.html#autotoc_md87", null ],
      [ "Optimisation, sparse systems and distances", "patterns.html#autotoc_md89", null ],
      [ "Estimation: tracking a moving thing", "patterns.html#autotoc_md91", null ],
      [ "Stochastic control and pricing", "patterns.html#autotoc_md93", null ],
      [ "Orderings, subsets and assignments", "patterns.html#autotoc_md95", null ],
      [ "Pipelines of stages", "patterns.html#autotoc_md97", null ],
      [ "Retrying", "patterns.html#autotoc_md99", null ],
      [ "Showcase: the whole library at work", "patterns.html#patterns_showcase", null ]
    ] ],
    [ "Topics", "topics.html", "topics" ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", "namespacemembers_dup" ],
        [ "Functions", "namespacemembers_func.html", "namespacemembers_func" ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Concepts", "concepts.html", "concepts" ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", "functions_vars" ],
        [ "Typedefs", "functions_type.html", "functions_type" ],
        [ "Enumerations", "functions_enum.html", null ],
        [ "Related Symbols", "functions_rela.html", "functions_rela" ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"accumulators_8hpp.html",
"classsnicholls_1_1containers_1_1ann__forest.html#a7c7cb0c29d4f581b73f4d01505c90990",
"classsnicholls_1_1containers_1_1md__span.html#ae0860ebb754c84b28312cb11fef7c7c9",
"classsnicholls_1_1game_1_1interventional__game.html#a72675ba77ef31d1a47eca15e2fed7fd8",
"classsnicholls_1_1linalg_1_1layout__stride_1_1mapping.html#a550c9d0242eb248a7e177a03a4fc775a",
"classsnicholls_1_1named__interface.html#a36d57cb836db80b2948bccb58b5c6cde",
"classsnicholls_1_1result__terminates_1_1result.html#a6fc565d9562b3a2bf77f37cb0481e34f",
"classsnicholls_1_1sql__connection__adapter.html#ac312c01561c5c32eadd361f7a18f9b7e",
"classsnicholls_1_1uint256.html#a64388d2ca01b5f676d84ffa0cf81fe0d",
"conceptsnicholls_1_1onramp__provider.html",
"group__core.html#ga00579d41dc20c2414e00edf8785ed8d6",
"group__core.html#ga2e0a3a47ecf33576dafd424d8a3b7346",
"group__core.html#ga5d2e253f46e63bb3a8138598b45c2da8",
"group__core.html#ga8a30427db04499ec77d4e40abb86d25b",
"group__core.html#gab7abca8b803bf65f3a1472bb4ddc107e",
"group__core.html#gae6c7aa596af8aa6017ccb928fdd5411d",
"group__core.html#gga224371c45ed55cc808c0072b60380b40a61a93ff236498725d515b0fe355161f2",
"group__core.html#gga997cdf403bc1fdb65d339be4800d8bacad1efad72dc5b17dc66a46767c32fff40",
"group__core.html#ggaf0f3cd725e9415f2d7350b869e159dada26eb25b14d930f9d5f59b2c50798a9a4",
"namespacesnicholls.html#a1e843645f34ba99d6ff192953c66cb96",
"namespacesnicholls_1_1fn_1_1types.html#a53ce063b7790a8a0481aa0f4959dcfeb",
"random_8hpp.html",
"structsnicholls_1_1constexpr__map.html",
"structsnicholls_1_1country_1_1MX.html",
"structsnicholls_1_1currency_1_1CYP.html",
"structsnicholls_1_1currency_1_1PLZ.html",
"structsnicholls_1_1err__in__place__t.html#ae19b3421f58b448e3addbc25da0bdb3e",
"structsnicholls_1_1fn__ptr_3_01R_07_5_08_07Args_8_8_8_08_4.html#ae7f543fa7d8b1da0b4702d5d6106e095",
"structsnicholls_1_1https__response.html#adb1e7c379ded7c62d2d5cdb59b0fa9b8",
"structsnicholls_1_1meta_1_1thread__all.html",
"structsnicholls_1_1ml_1_1block__mxfp4.html#aeb1a49260a529415bd96b290afda820b",
"structsnicholls_1_1ml_1_1gated__deltanet__weights.html#a733623000b68dd4374e026ba19c46998",
"structsnicholls_1_1ml_1_1mixer__activations.html#ab55d01c9eb20cee54d42f48ba2ac5b22",
"structsnicholls_1_1ml_1_1rank__head__weights.html#aa88cc04ca2efed397f6d4a089b0a7258",
"structsnicholls_1_1ml_1_1transformer__config.html#aa3f13b0e30c801d46c1529d90d02f6f9",
"structsnicholls_1_1payment.html#a52f30d58e935e85fa1530439813e6eef",
"structsnicholls_1_1send__receipt.html",
"structsnicholls_1_1stats_1_1bootstrap__distribution.html#ac6e58d53d8ad7b0280ef462948c6eb88",
"structsnicholls_1_1tracking_1_1bearing__range.html#a19804f0e0f87069af2d0bea347e8eb2d",
"type__list_8hpp.html"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
var LISTOFALLMEMBERS = 'List of all members';