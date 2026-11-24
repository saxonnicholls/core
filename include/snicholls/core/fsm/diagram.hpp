//
//  diagram.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `fsm::to_mermaid`, `to_dot`, `to_plantuml`: a machine's table drawn as Mermaid, Graphviz DOT or PlantUML,
/// generated from the table itself, so the documentation's diagram is the code.
/// \ingroup core
///
/// \overview A table is a specification; these turn it into the three diagram languages a reader is likely to have:
/// Mermaid (rendered by GitHub, GitLab and most documentation sites), Graphviz DOT, and PlantUML (sml's plant_uml
/// example, reproduced). Every row is drawn: initial states as `[*] -->`, `X` as `[*]`, guards in brackets, actions
/// after a slash, entry and exit actions and internal transitions on the state, composites as nested states,
/// orthogonal regions separated by `--`.
///
/// \snippet fsm/diagram_demo.cpp mermaid
///
/// State names come from the state: a `"name"_s` string, an enumerator's name, or the type's unqualified name.
/// Guards and actions are named by their types: a named function object reads well (`guard`, `send_ack`), a lambda is
/// drawn as `lambda`.
///
/// \relations `fsm/state_machine.hpp` (the table). Group: \ref core.
///
/// \impls `to_mermaid` (`stateDiagram-v2`), `to_dot` (a `digraph`, composites as clusters), `to_plantuml`
/// (`@startuml` ... `@enduml`, as sml's example writes it). Each takes the definition as a template argument or a
/// machine as an argument.
///
/// \contracts Pure functions of the table's type: the same table gives the same text on every platform, except for a
/// guard or action named by a compiler-spelled type. Not `constexpr` (they build a `std::string`).
///
/// \errors None. \throws std::bad_alloc
///
/// \examples \snippet fsm/state_machine_plant_uml_demo.cpp plant_uml
///
/// \perf Not benchmarked: run once, when documentation is generated.
///
/// \testing `tests/fsm/diagram_test.cpp`; `demos/fsm/diagram_demo.cpp` and `demos/fsm/state_machine_plant_uml_demo.cpp`.
///
/// \deps `<array>`, `<cstddef>`, `<cstdint>`, `<string>`, `<string_view>`, `<type_traits>`, and core's
/// `fsm/state_machine.hpp`.
///
/// \gotchas Mermaid draws orthogonal regions only inside a composite, so a machine with several regions is drawn as one
/// composite state named after the machine.
///
/// \status Experimental, new in 0.24.0.

#ifndef snicholls_core_fsm_diagram_hpp
#define snicholls_core_fsm_diagram_hpp

#include <snicholls/core/fsm/state_machine.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace snicholls::fsm {

    namespace detail {

        // ---- names of guards and actions ---------------------------------------------------------------------

        template <class F>
        struct callable_label {
            static std::string text()
            {
                const std::string_view name = short_name(type_name<F>());
                return name.find("lambda") != std::string_view::npos ? std::string("lambda") : std::string(name);
            }
        };
        template <class F>
            requires std::is_pointer_v<F> || std::is_member_function_pointer_v<F>
        struct callable_label<F> {
            static std::string text() { return "function"; }
        };
        template <>
        struct callable_label<always> {
            static std::string text() { return {}; }
        };
        template <>
        struct callable_label<none> {
            static std::string text() { return {}; }
        };
        template <>
        struct callable_label<defer_t> {
            static std::string text() { return "defer"; }
        };
        template <class E>
        struct callable_label<process_t<E>> {
            static std::string text() { return "process(" + std::string(event_info<E>::name) + ")"; }
        };
        template <class F>
        struct callable_label<wrapped<F>> : callable_label<F> {};
        template <class G>
        struct callable_label<not_<G>> {
            static std::string text() { return "!" + callable_label<G>::text(); }
        };
        template <class A, class B>
        struct callable_label<and_<A, B>> {
            static std::string text() { return callable_label<A>::text() + " && " + callable_label<B>::text(); }
        };
        template <class A, class B>
        struct callable_label<or_<A, B>> {
            static std::string text() { return callable_label<A>::text() + " || " + callable_label<B>::text(); }
        };
        template <class A, class B>
        struct callable_label<seq<A, B>> {
            static std::string text() { return callable_label<A>::text() + ", " + callable_label<B>::text(); }
        };
        template <class G, class A>
        struct callable_label<eval_t<G, A>> {
            static std::string text() { return "eval [" + callable_label<G>::text() + "] / " + callable_label<A>::text(); }
        };

        template <class A>
        inline constexpr bool is_seq_v = false;
        template <class A, class B>
        inline constexpr bool is_seq_v<seq<A, B>> = true;

        // "event [guard] / action", each part only when there is one.
        template <class R>
        std::string row_label()
        {
            std::string out(event_info<typename R::event_type>::name);
            const std::string guard = callable_label<typename R::guard_type>::text();
            const std::string action = callable_label<typename R::action_type>::text();
            if (!guard.empty())
                out += (out.empty() ? "[" : " [") + guard + "]";
            if (!action.empty()) {
                out += out.empty() ? "/ " : " / ";
                out += is_seq_v<typename R::action_type> ? "(" + action + ")" : action;
            }
            return out;
        }

        inline std::string diagram_id(std::string_view prefix, std::string_view name)
        {
            std::string out(prefix);
            for (const char c : name)
                out += ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) ? c : '_';
            return out;
        }

        inline std::string escaped(std::string_view text)
        {
            std::string out;
            for (const char c : text) {
                if (c == '"' || c == '\\')
                    out += '\\';
                out += c;
            }
            return out;
        }

        enum class diagram_language : std::uint8_t { mermaid, dot, plantuml };

        // One level of a machine, drawn: its rows in table order, its composites nested where they first appear.
        template <diagram_language Language, class Def>
        struct diagram_writer {
            using info = table_info<Def>;
            std::string& out;
            std::string prefix;
            std::string indent;

            template <class S>
            static constexpr bool composite = machine_definition<S>;

            template <class S>
            std::string id() const
            {
                if constexpr (std::is_same_v<S, terminate_state>)
                    return Language == diagram_language::dot ? prefix + "terminate" : "[*]";
                else
                    return diagram_id(prefix, state_info<S>::name);
            }

            void declare_names()
            {
                [&]<class... Ss>(meta::type_list<Ss...>) { (declare<Ss>(), ...); }(typename info::states{});
            }
            template <class S>
            void declare()
            {
                if constexpr (!std::is_same_v<S, terminate_state> && !composite<S>) {
                    const std::string_view name = state_info<S>::name;
                    if constexpr (Language == diagram_language::dot)
                        out += indent + id<S>() + " [label=\"" + escaped(name) + "\"];\n";
                    else if (diagram_id("", name) != name || !prefix.empty())
                        out += indent + "state \"" + escaped(name) + "\" as " + id<S>() + "\n";
                }
            }

            template <class S>
            void nested()
            {
                if constexpr (composite<S>) {
                    if (drawn_[info::template id_of<S>])
                        return;
                    drawn_[info::template id_of<S>] = true;
                    // PlantUML scopes a nested state's name, as sml's example relies on; Mermaid and DOT do not.
                    const std::string child = prefix + diagram_id("", state_info<S>::name) + "_";
                    const std::string inner_prefix = Language == diagram_language::plantuml ? prefix : child;
                    if constexpr (Language == diagram_language::dot) {
                        out += indent + "subgraph cluster_" + child + " {\n";
                        out += indent + "    label=\"" + escaped(state_info<S>::name) + "\";\n";
                    } else
                        out += indent + "state " + id<S>() + " {\n";
                    diagram_writer<Language, S> inner{out, inner_prefix, indent + "    "};
                    inner.draw();
                    out += indent + "}\n";
                }
            }

            // The node an edge to or from a composite attaches to in DOT: the start point of its cluster.
            template <class S>
            std::string endpoint() const
            {
                if constexpr (Language == diagram_language::dot && composite<S>)
                    return prefix + diagram_id("", state_info<S>::name) + "_start";
                else
                    return id<S>();
            }

            template <class R>
            void row(bool& initial_seen)
            {
                using Src = src_of<R>;
                using Dst = typename R::target;
                if constexpr (R::source::initial) {
                    if (initial_seen && Language != diagram_language::dot)
                        out += indent + "--\n";
                    initial_seen = true;
                    start<Src>();
                }
                nested<Src>();
                const std::string label = row_label<R>();
                if constexpr (std::is_same_v<Dst, internal_target>) {
                    if constexpr (Language == diagram_language::dot)
                        out += indent + endpoint<Src>() + " -> " + endpoint<Src>() + " [label=\"" + escaped(label) +
                            "\", style=dashed];\n";
                    else
                        out += indent + id<Src>() + " : " + label + "\n";
                } else {
                    nested<Dst>();
                    if constexpr (Language == diagram_language::dot) {
                        out += indent + endpoint<Src>() + " -> " + endpoint<Dst>() + " [";
                        if constexpr (composite<Src>)
                            out += "ltail=cluster_" + prefix + diagram_id("", state_info<Src>::name) + "_, ";
                        if constexpr (composite<Dst>)
                            out += "lhead=cluster_" + prefix + diagram_id("", state_info<Dst>::name) + "_, ";
                        out += "label=\"" + escaped(label) + "\"];\n";
                    } else {
                        out += indent + id<Src>() + " --> " + id<Dst>();
                        if (!label.empty())
                            out += " : " + label;
                        out += "\n";
                    }
                }
            }

            template <class S>
            void start()
            {
                if constexpr (Language == diagram_language::dot) {
                    const std::string point = prefix + "start" + std::to_string(starts_++);
                    out += indent + point + " [shape=point];\n";
                    out += indent + point + " -> " + endpoint<S>() + (composite<S> ? " [lhead=cluster_" + prefix +
                                                                                      diagram_id("", state_info<S>::name) + "_];\n"
                                                                                : std::string(";\n"));
                } else
                    out += indent + "[*] --> " + id<S>() + "\n";
            }

            void draw()
            {
                if constexpr (Language == diagram_language::dot) {
                    out += indent + prefix + "start [shape=point, style=invis];\n";
                    if constexpr (info::terminal_id != npos)
                        out += indent + prefix + "terminate [shape=doublecircle, label=\"\"];\n";
                }
                declare_names();
                bool initial_seen = false;
                [&]<class... Rs>(meta::type_list<Rs...>) { (row<Rs>(initial_seen), ...); }(typename info::rows{});
            }

            std::array<bool, info::state_count> drawn_{};
            std::size_t starts_ = 0;
        };

        template <class T>
        struct definition_of {
            using type = T;
        };
        template <class Def, class... Ps>
        struct definition_of<sm<Def, Ps...>> {
            using type = Def;
        };
    } // namespace detail

    /// \brief The table as a Mermaid state diagram (`stateDiagram-v2`).
    /// \ingroup core
    /// \tparam Def the definition, or an `sm<Def, ...>` type.
    /// \return the diagram.
    /// \throws std::bad_alloc
    template <class Def>
    [[nodiscard]] std::string to_mermaid()
    {
        using D = typename detail::definition_of<Def>::type;
        std::string out = "stateDiagram-v2\n";
        if constexpr (detail::table_info<D>::regions > 1) {
            const std::string name = detail::diagram_id("", detail::short_name(type_name<D>()));
            out += "    state " + name + " {\n";
            detail::diagram_writer<detail::diagram_language::mermaid, D>{out, name + "_", "        "}.draw();
            out += "    }\n    [*] --> " + name + "\n";
        } else
            detail::diagram_writer<detail::diagram_language::mermaid, D>{out, "", "    "}.draw();
        return out;
    }

    /// \brief The table as a Graphviz `digraph`, composites as clusters.
    /// \ingroup core
    /// \tparam Def the definition, or an `sm<Def, ...>` type.
    /// \return the diagram.
    /// \throws std::bad_alloc
    template <class Def>
    [[nodiscard]] std::string to_dot()
    {
        using D = typename detail::definition_of<Def>::type;
        std::string out = "digraph " + detail::diagram_id("", detail::short_name(type_name<D>())) + " {\n";
        out += "    rankdir=LR;\n    compound=true;\n    node [shape=box, style=rounded];\n";
        detail::diagram_writer<detail::diagram_language::dot, D>{out, "", "    "}.draw();
        out += "}\n";
        return out;
    }

    /// \brief The table as PlantUML, as sml's plant_uml example writes it.
    /// \ingroup core
    /// \tparam Def the definition, or an `sm<Def, ...>` type.
    /// \return the diagram.
    /// \throws std::bad_alloc
    template <class Def>
    [[nodiscard]] std::string to_plantuml()
    {
        using D = typename detail::definition_of<Def>::type;
        std::string out = "@startuml\n\n";
        detail::diagram_writer<detail::diagram_language::plantuml, D>{out, "", ""}.draw();
        out += "\n@enduml\n";
        return out;
    }

    /// \brief `to_mermaid` of a machine's table.
    /// \ingroup core
    /// \return the diagram.
    /// \throws std::bad_alloc
    template <class Def, class... Ps>
    [[nodiscard]] std::string to_mermaid(const sm<Def, Ps...>&)
    {
        return to_mermaid<Def>();
    }

    /// \brief `to_dot` of a machine's table.
    /// \ingroup core
    /// \return the diagram.
    /// \throws std::bad_alloc
    template <class Def, class... Ps>
    [[nodiscard]] std::string to_dot(const sm<Def, Ps...>&)
    {
        return to_dot<Def>();
    }

    /// \brief `to_plantuml` of a machine's table.
    /// \ingroup core
    /// \return the diagram.
    /// \throws std::bad_alloc
    template <class Def, class... Ps>
    [[nodiscard]] std::string to_plantuml(const sm<Def, Ps...>&)
    {
        return to_plantuml<Def>();
    }

} // namespace snicholls::fsm

#endif /* snicholls_core_fsm_diagram_hpp */
