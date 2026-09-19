# Code style

[中文](../zh/code-style.md)

These conventions keep Python and TileLang source concise and consistent.

## Imports

Use original import names. The TileLang DSL follows `import tilelang.language as T`.
Import classes and types from their defining modules; call functions through their
containing modules. Keep package initializers lightweight and load dependencies
close to use.

## Naming and types

Use mathematical names for reusable operations and model-specific names for model
semantics. Kernel symbols combine the algorithm name with workload and launch
parameters. Use snake_case for modules and functions and PascalCase for classes.

Separate workload dimensions (`m`, `n`, `k`) from tile dimensions (`tile_m`,
`tile_n`, `tile_k`). Use typed records for configuration and results and immutable
records for metadata. Annotate public interfaces precisely. Required extension
points use `ABC` and `abstractmethod`.

## Command-line interfaces

Express command-line parameters, options and subcommands with Typer. Keep parsing
and validation in typed command functions instead of manually reading `sys.argv`.

## Assertions

Express internal invariants directly with `assert condition, "message"`. Do not
expand an assertion into `if not condition: raise ...`. Exceptions remain for
failures caused by external input, files, devices, or other runtime conditions.

## TileLang

Keep one JIT definition per kernel implementation file. Express execution variants
as compile-time parameters. Explain numerical choices and hardware layouts with
concise comments. Prefer direct data flow and reuse loaded operands where measured
execution benefits.
