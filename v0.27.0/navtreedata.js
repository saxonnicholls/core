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
      [ "Tokenizer throughput", "index.html#autotoc_md3", null ]
    ] ],
    [ "Using it", "index.html#autotoc_md4", [
      [ "Two copies in one build", "index.html#autotoc_md5", null ]
    ] ],
    [ "Building and testing", "index.html#autotoc_md6", null ],
    [ "Licence", "index.html#autotoc_md7", null ],
    [ "snicholls::result - Rust's <span class=\"tt\">Result&lt;V, E&gt;</span>, zero overhead, rethrowable, with traced errors", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html", [
      [ "Decisions", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md9", null ],
      [ "Per-call choices", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md10", null ],
      [ "API at a glance", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md11", null ],
      [ "Layout guarantees (tests/result_test.cpp)", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md12", null ],
      [ "Performance", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md13", null ],
      [ "Interop", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md14", null ],
      [ "Open items", "md__2home_2runner_2work_2core_2core_2RESULT__PLAN.html#autotoc_md15", null ]
    ] ],
    [ "Patterns", "patterns.html", [
      [ "Principles behind every pattern", "patterns.html#autotoc_md16", null ],
      [ "Errors", "patterns.html#autotoc_md18", [
        [ "Expected failure is a value: <span class=\"tt\">result&lt;V, E&gt;</span>", "patterns.html#autotoc_md19", null ],
        [ "Errors for logs: <span class=\"tt\">traced&lt;E&gt;</span>", "patterns.html#autotoc_md20", null ]
      ] ],
      [ "Capabilities: a contract, its twin, a reference and a kit", "patterns.html#autotoc_md22", null ],
      [ "State machines", "patterns.html#autotoc_md24", [
        [ "A machine is a table", "patterns.html#autotoc_md25", null ],
        [ "Save, load and draw it", "patterns.html#autotoc_md26", null ],
        [ "Lifecycles that read as the domain", "patterns.html#autotoc_md27", null ],
        [ "Machines, functionally", "patterns.html#autotoc_md28", null ]
      ] ],
      [ "Presentation is an attribute of the type", "patterns.html#autotoc_md30", null ],
      [ "Printing", "patterns.html#autotoc_md32", [
        [ "One message, one write", "patterns.html#autotoc_md33", null ],
        [ "A type that prints itself", "patterns.html#autotoc_md34", null ]
      ] ],
      [ "Dispatch over types and values", "patterns.html#autotoc_md36", [
        [ "Overload sets, never <span class=\"tt\">if constexpr</span> chains", "patterns.html#autotoc_md37", null ],
        [ "Values with <span class=\"tt\">constexpr_for</span>, types with <span class=\"tt\">type_list</span>", "patterns.html#autotoc_md38", null ],
        [ "Arguments that exist only at run time", "patterns.html#autotoc_md39", null ]
      ] ],
      [ "Reflection over plain aggregates", "patterns.html#autotoc_md41", null ],
      [ "Tables built by the compiler", "patterns.html#autotoc_md43", null ],
      [ "Exact numbers, time and identity", "patterns.html#autotoc_md45", [
        [ "Money is exact", "patterns.html#autotoc_md46", null ],
        [ "Time is explicit", "patterns.html#autotoc_md47", null ],
        [ "Identity is a type", "patterns.html#autotoc_md48", null ],
        [ "Secrets do not leak", "patterns.html#autotoc_md49", null ]
      ] ],
      [ "Dated money: one cash flow", "patterns.html#autotoc_md51", null ],
      [ "Statistics that merge", "patterns.html#autotoc_md53", [
        [ "Randomness that reproduces", "patterns.html#autotoc_md54", null ]
      ] ],
      [ "Attribution: who moved the number", "patterns.html#autotoc_md56", null ],
      [ "Functional programming, at compile time", "patterns.html#autotoc_md58", [
        [ "Functions that return functions", "patterns.html#autotoc_md59", null ],
        [ "Folds, unfolds and laziness", "patterns.html#autotoc_md60", null ],
        [ "Type classes and effects", "patterns.html#autotoc_md61", null ],
        [ "OCaml's pipes and functors", "patterns.html#autotoc_md62", null ],
        [ "Parsers from combinators", "patterns.html#autotoc_md63", null ],
        [ "Classics, written in C++", "patterns.html#autotoc_md64", null ],
        [ "An interpreter: algebraic data types and pattern matching", "patterns.html#autotoc_md65", null ],
        [ "Effects: railway, state, reader and writer", "patterns.html#autotoc_md66", null ],
        [ "Rust's vocabulary", "patterns.html#autotoc_md67", null ],
        [ "Rewriting and point-wise functions", "patterns.html#autotoc_md68", null ]
      ] ],
      [ "Signals at compile time", "patterns.html#autotoc_md70", null ],
      [ "Fixed sizes for the device", "patterns.html#autotoc_md72", null ],
      [ "Containers keyed by identity", "patterns.html#autotoc_md74", null ],
      [ "Hashing: the right hash for the job", "patterns.html#autotoc_md76", null ],
      [ "Linear algebra: one interface, a backend chosen by type", "patterns.html#autotoc_md78", null ],
      [ "Machine learning as small, composable kernels", "patterns.html#autotoc_md80", null ],
      [ "Optimisation, sparse systems and distances", "patterns.html#autotoc_md82", null ],
      [ "Estimation: tracking a moving thing", "patterns.html#autotoc_md84", null ],
      [ "Stochastic control and pricing", "patterns.html#autotoc_md86", null ],
      [ "Orderings, subsets and assignments", "patterns.html#autotoc_md88", null ],
      [ "Pipelines of stages", "patterns.html#autotoc_md90", null ],
      [ "Retrying", "patterns.html#autotoc_md92", null ],
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
"classsnicholls_1_1containers_1_1ann__forest.html#ace0ef94bd3e559ef8340e74fe7e76ee6",
"classsnicholls_1_1containers_1_1md__span.html#ae19f030d025d73d35fa208eec2400fa9",
"classsnicholls_1_1game_1_1interventional__game.html#af4ecc1a205526dc3da86c6a83b1213d8",
"classsnicholls_1_1linalg_1_1layout__stride_1_1mapping.html#a64acedcabbaa4cec72f3fc2130f256b6",
"classsnicholls_1_1named__model.html",
"classsnicholls_1_1result__terminates_1_1result.html#a736fba1a37f4a0fb284797d9afdf4df0",
"classsnicholls_1_1sql__connection__adapter.html#acb538774bd757696bd2cfe612cf6e301",
"classsnicholls_1_1uint256.html#a658ad69c655084f901c68bfb742ce72e",
"conceptsnicholls_1_1owning__string.html",
"group__core.html#ga00721fdda55cc291f6ec9bbe957dd874",
"group__core.html#ga2e0bb69059013fe5cacf0f5728d8aa5b",
"group__core.html#ga5d4999acd25a80a2ff036cb345f21e8d",
"group__core.html#ga8a6868c18e99eb39d92b41d3e0ca56b4",
"group__core.html#gab7adced5013d6d3b98fc2e451dababa6",
"group__core.html#gae6cee736b5b6b7072adf55a068022af8",
"group__core.html#gga224371c45ed55cc808c0072b60380b40a6f8be0553c0aa8e459f00b113b17c8a0",
"group__core.html#gga997cdf403bc1fdb65d339be4800d8bacae55d43eabeefe5a8271b4a3c898bd18f",
"group__core.html#ggaf0f3cd725e9415f2d7350b869e159dada8e13ffc9fd9d6a6761231a764bdf106b",
"namespacesnicholls.html#a3598a673b0a57cbc560854b5fc76b86f",
"namespacesnicholls_1_1literals.html#aa03ed4d80504c25441c8278aac460327",
"role__store__reference_8hpp.html",
"structsnicholls_1_1containers_1_1ann__default__split_3_01metric_1_1angular_01_4.html#adf013335175a7f8d09b739fd81d31f0c",
"structsnicholls_1_1country_1_1PH.html",
"structsnicholls_1_1currency_1_1EEK.html",
"structsnicholls_1_1currency_1_1RUR.html",
"structsnicholls_1_1error__report.html#af41ae348db0cae5eb4fdad579d8b16bb",
"structsnicholls_1_1fsm_1_1dealing__event_1_1price.html",
"structsnicholls_1_1inplace__string.html#adbc155128dac8bd5a8037096b7e3a68a",
"structsnicholls_1_1metric_1_1chebyshev.html#a875b66a5c3a305ee5979fb7acf8d6d42",
"structsnicholls_1_1ml_1_1block__q3__k.html#a237594cc012da887578e7f015c08008f",
"structsnicholls_1_1ml_1_1gla__gradients.html#a87f2c93c2213b3fee716fd8a113e5c12",
"structsnicholls_1_1ml_1_1mlp__gradients.html#a351c14a3eecca0fe6f2b6890a83fe882",
"structsnicholls_1_1ml_1_1rope__mode__table.html#a042dd0cb24ea2e9f73bb5a74d38f09ce",
"structsnicholls_1_1ml_1_1transformer__config.html#aec01774d0494cb03415b3c63aa391038",
"structsnicholls_1_1payment__fact.html#afaaee46cfd97dedd384fdce011af379c",
"structsnicholls_1_1signal_1_1biquad.html#a0122ef66ab0b570ac31cc39f0718134c",
"structsnicholls_1_1stats_1_1collect.html#a9c006e0f686d4c2b9e3c6ba861aaf38a",
"structsnicholls_1_1tracking_1_1combined.html#ab08cc206d7a4c299df3366fdb80a7f45",
"week__date_8hpp.html"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
var LISTOFALLMEMBERS = 'List of all members';