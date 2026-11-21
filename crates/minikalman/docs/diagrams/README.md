# Diagrams

The mermaid diagrams in this directory are included into the rustdoc pages with the crate's
internal `diagram!` macro, which appends `class <node> focus` lines to highlight the part a
page documents. GitHub renders mermaid natively; on docs.rs, `docs/mermaid.html` renders it.

## Visual language

Every block diagram uses the same conventions, so a reader can move between pages:

| Element | Shape | Mermaid |
|---|---|---|
| Filter state (estimate, covariance) | stadium | `id(["x̂, P"])` |
| External input (control, measurement, callback) | parallelogram | `id[/"z"/]` |
| Computation step | rectangle, name then formulas | `id["predict<br/>x̂ ← A·x̂"]` |
| Check (gate, convergence) | hexagon | `id{{"gate"}}` |
| Outcome (`CorrectionStatus`) | rounded rectangle | `id("Applied")` |

- **Flow:** left to right (`flowchart LR`), one filter step from `x̂ₖ₋₁` to `x̂ₖ`, without a cycle edge (it tangles the layout). Solid arrows carry data on every run; dotted
  arrows are optional inputs or early exits, labelled with their condition.
- **Notation:** the crate's symbols: `x̂`, `P`, `A`, `B`, `Q`, `Qᵤ`, `H`, `R`, `y`, `S`, `K`,
  `u`, `z`. Assignments use `←`, definitions `=`, products `·`.
- **Highlight:** only the `focus` class, which sets fill, stroke and text color together so
  that it stays readable in light and dark themes. No other colors.
- **Node IDs** are stable per diagram: `prior`, `predict`, `control`, `correct`, `posterior`, and in the
  correction zoom `innovation`, `covariance`, `gate`, `gain`, `update`, `adapt`.
- Keep diagrams at about 15 nodes or fewer, and use classic flowchart syntax only (quoted
  labels, `<br/>`), since GitHub's mermaid lags behind docs.rs.
