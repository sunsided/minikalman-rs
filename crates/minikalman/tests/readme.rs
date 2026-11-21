//! Checks on the README's Rust code blocks, which run as doctests but render as plain Markdown.

#![forbid(unsafe_code)]

/// The README, as rendered on GitHub and crates.io.
const README: &str = include_str!("../../../README.md");

/// The statically allocated README example, also run as a doctest without `alloc`.
const NO_ALLOC_EXAMPLE: &str = include_str!("../docs/readme-no-alloc.md");

/// Rustdoc hides `# ` lines, but GitHub shows them, so README code must not rely on them.
#[test]
fn rust_blocks_have_no_hidden_lines() {
    let mut in_rust = false;
    let mut hidden = Vec::new();
    for (number, line) in README.lines().enumerate() {
        if line.starts_with("```") {
            in_rust = line.starts_with("```rust");
            continue;
        }

        if in_rust && (line == "#" || line.starts_with("# ")) {
            hidden.push(format!("README.md:{}: {line}", number + 1));
        }
    }

    assert!(
        hidden.is_empty(),
        "hidden doctest lines:\n{}",
        hidden.join("\n")
    );
}

/// The README must show the same no-alloc example that the `ReadmeNoAllocDoctest` runs.
#[test]
fn no_alloc_example_matches_its_doctest() {
    assert!(
        README.contains(NO_ALLOC_EXAMPLE),
        "README.md must contain crates/minikalman/docs/readme-no-alloc.md verbatim"
    );
}
