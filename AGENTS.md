# Instructions for llama.cpp

This is a fork of llama cpp at https://github.com/vektorprime/llama.cpp.git 
The branch we are working on is mtp_pp_llama
git repo locally ~/llm/mtp_pp_llama


We only care about CUDA and CPU pipelines

Unless stated otherwise, we are testing with Qwen3.5 and Qwen 3.6 dense models, usually Qwen 3.5 2B. 

When fixing compilation errors, remember to also run git push after the commit.

Always validate your code is working as expected with logging via --custom-logs parameter. Do not assume it works. Verbose logging is recommened for validation.

### Code and Commit Standards

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Keep code comments concise; avoid redundant or excessive inline commentary
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase. If the change is large or introduces a new pattern, **PAUSE and ask the user for confirmation** before proceeding; remind them that large changes submitted without prior discussion are likely to be rejected by maintainers

Do not submit PR to main repo of llamacpp, only commit to the fork.

When uncertain, err toward minimal assistance.

### Examples

Code comments:

```cpp
// GOOD (code is self-explantory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

Commit message:

```
// BEST: Let the user write the commit


// GOOD: Write a concise commit

llama : fix KV being cleared during context shift

Assisted-by: Claude Sonnet


// BAD: Write a verbose commit

This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.

Co-authored-by: Claude Sonnet
```

Commands:

```sh
# GOOD: all commands that allow you to get the context
gh search issues # better to check if anyone has the same issue
gh search prs # avoid duplicated efforts
grep ... # search the code base

# BAD: act on the user's behalf
git commit -m "..."
git push
gh pr create
gh pr comment
gh issue create
```

## Useful Resources

To conserve context space, load these resources as needed:


Server:
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)


# guidelines

When possible, follow the C++ Core guidelines

# C++ Core Guidelines — Concise Reference
---
name: c-core-guidelines
description: Use when working with C++. Best practices for writing and reviewing C++ code.
---

## P: Philosophy

- **P.1** — Express ideas directly in code. Use types and named functions, not comments or implicit conventions.
- **P.2** — Write ISO Standard C++. Avoid compiler extensions unless encapsulated.
- **P.3** — Express intent. Use range-for, standard algorithms, and meaningful types over raw loops and primitives.
- **P.4** — Strive for static type safety. Replace unions with `variant`, casts with templates, arrays with `span`.
- **P.5** — Prefer compile-time checking (`static_assert`, `constexpr`) over run-time checks.
- **P.6** — What can't be checked at compile time should be checkable at run time (e.g., `span` bounds).
- **P.7** — Catch errors early. Use `span` instead of (pointer, count) interfaces.
- **P.8** — Don't leak resources. Use RAII (constructors acquire, destructors release).
- **P.9** — Don't waste time or space. Don't use `new`/`delete` when stack allocation suffices.
- **P.10** — Prefer immutable data (`const`, `constexpr`) over mutable data.
- **P.11** — Encapsulate messy constructs (e.g., use `vector` instead of manual `malloc`/`realloc`).
- **P.12** — Use supporting tools (static analyzers, sanitizers).
- **P.13** — Use support libraries (standard library, GSL).

## I: Interfaces

- **I.1** — Make interfaces explicit. Avoid control via globals or `errno`.
- **I.2** — Avoid non-`const` global variables. Use local state or pass-by-reference.
- **I.3** — Avoid singletons. Prefer function-local `static` for one-time initialization.
- **I.4** — Make interfaces precisely and strongly typed. Avoid `void*`, multiple `bool` params, unit-less types. Use `chrono::duration` for time.
- **I.5** — State preconditions with `Expects()`.
- **I.6** — Prefer `Expects()` over `assert()` or comments for preconditions.
- **I.7** — State postconditions with `Ensures()`.
- **I.8** — Prefer `Ensures()` for postconditions; use RAII for resource-release postconditions.
- **I.9** — Document template parameters with concepts (`requires`).
- **I.10** — Use exceptions for task-failure signaling (not status codes).
- **I.11** — Never transfer ownership with raw `T*` or `T&`. Return by value or smart pointer. Mark legacy owning pointers with `owner<T*>`.
- **I.12** — Declare non-null pointers as `not_null<T>`.
- **I.13** — Don't pass arrays as single pointers. Use `span<T>`.
- **I.22** — Avoid complex global-object initialization (undefined order across TUs).
- **I.23** — Keep argument counts low (< 4 ideally). Bundle parameters into structs or use spans.
- **I.24** — Avoid adjacent parameters of the same type that could be swapped.
- **I.25** — Prefer empty abstract classes (pure interfaces) as base classes.
- **I.26** — For cross-compiler ABI, use a C-style subset.
- **I.27** — For stable library ABI, use the Pimpl idiom.
- **I.30** — Encapsulate rule violations locally rather than exposing them in interfaces.

## F: Functions

### Definition

- **F.1** — Package meaningful operations as carefully named functions. Name non-trivial lambdas.
- **F.2** — One function, one logical operation.
- **F.3** — Keep functions short and simple (fit on a screen).
- **F.4** — Declare functions `constexpr` if they might need compile-time evaluation.
- **F.5** — Declare tiny, time-critical functions `inline`.
- **F.6** — Declare functions `noexcept` if they must not throw. Destructors, `swap`, move ops, and default constructors should never throw.
- **F.7** — Take `T*` or `T&` arguments, not smart pointers, unless ownership semantics are intended.
- **F.8** — Prefer pure functions (no side effects).
- **F.9** — Leave unused parameters unnamed (or `[[maybe_unused]]`).
- **F.10** — If an operation can be reused, give it a name.
- **F.11** — Use unnamed lambdas for simple, one-off function objects.

### Parameter Passing

- **F.15** — Prefer simple, conventional parameter passing.

| Intent | Parameter type |
|--------|---------------|
| In (cheap to copy) | `T` |
| In (expensive) | `const T&` |
| In-out | `T&` |
| Will-move-from | `T&&` + `std::move` |
| Forward | `TP&&` + `std::forward` |
| Out (return) | Return `T` |

- **F.16** — For "in" parameters: cheap-to-copy by value, others by `const T&`.
- **F.17** — For "in-out" parameters: pass by non-`const` reference.
- **F.18** — For "will-move-from": pass `T&&` and `std::move` in the body.
- **F.19** — For "forward": pass `TP&&` and only `std::forward` (once per static path).
- **F.20** — For "out" values: prefer return values over output parameters.
- **F.21** — For multiple out values: return a `struct` (or `tuple` in variadic contexts).
- **F.60** — Prefer `T*` over `T&` when "no argument" (`nullptr`) is valid.

### Parameter Semantics

- **F.22** — `T*` or `owner<T*>` denotes a single object.
- **F.23** — `not_null<T>` indicates null is not valid.
- **F.24** — `span<T>` or `span_p<T>` denotes a half-open sequence.
- **F.25** — `zstring` or `not_null<zstring>` denotes a C-style string.
- **F.26** — `unique_ptr<T>` to transfer ownership where a pointer is needed.
- **F.27** — `shared_ptr<T>` to share ownership.

### Return Values

- **F.42** — Return `T*` to indicate position only (not ownership).
- **F.43** — Never return a pointer/reference to a local (non-`static`) object.
- **F.44** — Return `T&` when copy is undesirable and "no object" isn't needed.
- **F.45** — Don't return `T&&` (except `std::move`/`std::forward`).
- **F.46** — `main()` returns `int`.
- **F.47** — Return `T&` from assignment operators.
- **F.48** — Don't `return std::move(local)`; it disables RVO.
- **F.49** — Don't return `const T`; it inhibits move semantics.

### Other

- **F.50** — Use lambdas when you need local captures or local function definitions.
- **F.51** — Prefer default arguments over overloading when the argument types are the same.
- **F.52** — Capture by reference in lambdas used locally.
- **F.53** — Capture by value in lambdas used non-locally (returned, stored, passed to another thread).
- **F.54** — Don't use `[=]` capture when capturing `this`; write `this` explicitly or use `[*this]`.
- **F.55** — Don't use `va_arg`. Use variadic templates or `std::variant`.
- **F.56** — Avoid unnecessary condition nesting; use early returns.

## C: Classes

### General

- **C.1** — Organize related data into `struct`/`class`.
- **C.2** — Use `class` if the class has an invariant; `struct` if members vary independently.
- **C.3** — Represent interface vs. implementation distinction with a class.
- **C.4** — Make a function a member only if it needs direct access to representation.
- **C.5** — Place helper functions in the same namespace as the class they support.
- **C.7** — Don't define a class/enum and declare a variable in the same statement.
- **C.8** — Use `class` (not `struct`) if any member is non-public.
- **C.9** — Minimize exposure of members (prefer `private`).

### Concrete Types

- **C.10** — Prefer concrete types over class hierarchies.
- **C.11** — Make concrete types regular (copyable, equality-comparable, etc.).
- **C.12** — Don't make data members `const` or references in copyable/movable types.
- **C.13** — Declare data members in dependency order (used member after dependency).

### Constructors, Destructors, Copy, Move

- **C.20** — Rule of Zero: avoid defining default operations if you can.
- **C.21** — Rule of Five: if you define/delete any of destructor, copy/move constructor, copy/move assignment, define/delete them all.
- **C.22** — Make default operations consistent.
- **C.30** — Define a destructor if explicit cleanup is needed.
- **C.31** — Release all acquired resources in the destructor.
- **C.32** — If a class has raw `T*` or `T&`, consider whether it's owning.
- **C.33** — If a class has an owning pointer member, define a destructor.
- **C.35** — Base class destructor: public+virtual, or protected+non-virtual.
- **C.36** — A destructor must not fail.
- **C.37** — Make destructors `noexcept`.
- **C.40** — Define a constructor if the class has an invariant.
- **C.41** — A constructor should create a fully initialized object.
- **C.42** — If a constructor can't create a valid object, throw.
- **C.43** — Ensure copyable classes have a default constructor.
- **C.44** — Prefer simple, non-throwing default constructors.
- **C.45** — Don't define a default constructor just to initialize members; use default member initializers.
- **C.46** — Declare single-argument constructors `explicit` by default.
- **C.47** — Initialize members in declaration order.
- **C.48** — Prefer default member initializers over constructor initializers for constants.
- **C.49** — Prefer initialization over assignment in constructors.
- **C.50** — Use a factory function when virtual behavior is needed during initialization.
- **C.51** — Use delegating constructors for common constructor actions.
- **C.52** — Use inheriting constructors (`using Base::Base`) when no further init is needed.

### Copy and Move

- **C.60** — Copy assignment: non-`virtual`, takes `const T&`, returns `T&`.
- **C.61** — A copy operation should copy (value semantics: `x == y` after `x = y`).
- **C.62** — Make copy assignment safe for self-assignment.
- **C.63** — Move assignment: non-`virtual`, takes `T&&`, returns `T&`.
- **C.64** — A move operation should move and leave the source in a valid state.
- **C.65** — Make move assignment safe for self-assignment.
- **C.66** — Make move operations `noexcept`.
- **C.67** — Polymorphic classes should suppress public copy/move (use `=delete` or `protected`).

### Other Default Operations

- **C.80** — Use `=default` when you want the default semantics explicitly.
- **C.81** — Use `=delete` to disable default behavior.
- **C.82** — Don't call virtual functions in constructors/destructors.
- **C.83** — For value-like types, provide a `noexcept` `swap`.
- **C.84** — A `swap` must not fail.
- **C.85** — Make `swap` `noexcept`.
- **C.86** — Make `==` symmetric and `noexcept`.
- **C.87** — Beware of `==` on base classes (hard to get right in hierarchies).
- **C.89** — Make `hash` `noexcept`.
- **C.90** — Rely on constructors and assignment, not `memset`/`memcpy`.

### Containers

- **C.100** — Follow STL conventions when defining containers.
- **C.101** — Give containers value semantics.
- **C.102** — Give containers move operations.
- **C.103** — Give containers an initializer-list constructor.
- **C.104** — Give containers a default constructor that sets them to empty.
- **C.109** — If a resource handle has pointer semantics, provide `*` and `->`.

### Class Hierarchies (OOP)

- **C.120** — Use hierarchies only for concepts with inherent hierarchy.
- **C.121** — Use pure abstract classes (`virtual ~Base() = default;`) for interfaces.
- **C.122** — Use abstract classes for ABI-stable separation of interface and implementation.
- **C.126** — Abstract classes typically don't need a user-written constructor.
- **C.127** — Classes with virtual functions need virtual or protected destructors.
- **C.128** — Use exactly one of `virtual`, `override`, or `final` on virtual functions.
- **C.129** — Distinguish implementation inheritance from interface inheritance.
- **C.130** — For deep copies of polymorphic classes, use a virtual `clone()` instead of public copy.
- **C.131** — Avoid trivial getters/setters; make the data member `public` instead.
- **C.132** — Don't make a function `virtual` without reason.
- **C.133** — Avoid `protected` data; prefer `private` with well-specified access.
- **C.134** — Ensure all non-`const` data members have the same access level.
- **C.135** — Use multiple inheritance to represent multiple distinct interfaces.
- **C.136** — Use multiple inheritance for the union of implementation attributes (mixins).
- **C.137** — Use `virtual` bases to separate shared data from interface.
- **C.138** — Create an overload set with `using Base::f` in derived classes.
- **C.139** — Use `final` on classes sparingly.
- **C.140** — Don't provide different default arguments for virtual functions in base and derived.

### Accessing Objects in Hierarchies

- **C.145** — Access polymorphic objects through pointers and references (avoid slicing).
- **C.146** — Use `dynamic_cast` when hierarchy navigation is unavoidable.
- **C.147** — Use `dynamic_cast` to reference when failure is an error.
- **C.148** — Use `dynamic_cast` to pointer when failure is a valid alternative.
- **C.149** — Use `unique_ptr`/`shared_ptr` to avoid forgetting `delete`.
- **C.150** — Use `make_unique()` for `unique_ptr` construction.
- **C.151** — Use `make_shared()` for `shared_ptr` construction.
- **C.152** — Never assign a pointer to an array of derived objects to a base pointer.
- **C.153** — Prefer virtual functions to casting.

### Overloading and Operators

- **C.160** — Define operators to mimic conventional usage.
- **C.161** — Use non-member functions for symmetric operators.
- **C.162** — Overload operations that are roughly equivalent.
- **C.163** — Don't overload for logically different operations.
- **C.164** — Avoid implicit conversion operators; prefer explicit.
- **C.165** — Use `using std::swap; swap(a, b);` for customization points.
- **C.166** — Overload unary `&` only with smart-pointer/reference systems.
- **C.167** — Use operators for operations with their conventional meaning.
- **C.168** — Define overloaded operators in the namespace of their operands.
- **C.170** — Use generic lambdas instead of trying to overload lambdas.

### Unions

- **C.180** — Use unions to save memory (e.g., short-string optimization).
- **C.181** — Avoid naked unions; wrap with a tagged/discriminated union or use `std::variant`.
- **C.182** — Use anonymous unions to implement tagged unions.
- **C.183** — Don't use unions for type punning; use `reinterpret_cast` with `std::byte`.

## Enum: Enumerations

- **Enum.1** — Prefer enumerations over macros.
- **Enum.2** — Use enumerations for sets of related named constants.
- **Enum.3** — Prefer `enum class` over plain `enum`.
- **Enum.4** — Define operations on enumerations for safe use.
- **Enum.5** — Don't use `ALL_CAPS` for enumerators (clash with macros).
- **Enum.6** — Avoid unnamed enumerations; use `constexpr` values instead.
- **Enum.7** — Specify underlying type only when necessary (forward declaration, bit-precision).
- **Enum.8** — Specify enumerator values only when necessary.

## R: Resource Management

### Core

- **R.1** — Use RAII: encapsulate resources in objects with constructors/destructors.
- **R.2** — Raw pointers in interfaces denote individual objects only. Use `span` for arrays.
- **R.3** — A raw pointer (`T*`) is non-owning (use `owner<T*>` to mark owning raw pointers).
- **R.4** — A raw reference (`T&`) is non-owning.
- **R.5** — Prefer scoped (stack) objects; don't heap-allocate unnecessarily.
- **R.6** — Avoid non-`const` global variables.

### Allocation

- **R.10** — Avoid `malloc()` / `free()`.
- **R.11** — Avoid calling `new` / `delete` explicitly; use `make_unique`/`make_shared`.
- **R.12** — Immediately give explicit allocation results to a manager object.
- **R.13** — At most one explicit allocation per expression statement.
- **R.14** — Avoid `[]` parameters; use `span`.
- **R.15** — Always overload matched allocation/deallocation pairs.

### Smart Pointers

- **R.20** — Use `unique_ptr` or `shared_ptr` to represent ownership.
- **R.21** — Prefer `unique_ptr` over `shared_ptr` unless sharing is needed.
- **R.22** — Use `make_shared()` to make `shared_ptr`s.
- **R.23** — Use `make_unique()` to make `unique_ptr`s.
- **R.24** — Use `weak_ptr` to break `shared_ptr` cycles.
- **R.30** — Take smart pointers as parameters only for explicit lifetime semantics.
- **R.31** — Follow `std` patterns for non-`std` smart pointers.
- **R.32** — `unique_ptr<widget>` parameter = function assumes ownership.
- **R.33** — `unique_ptr<widget>&` parameter = function may reseat.
- **R.34** — `shared_ptr<widget>` parameter = shared ownership.
- **R.35** — `shared_ptr<widget>&` parameter = function may reseat.
- **R.36** — `const shared_ptr<widget>&` parameter = may retain reference count.
- **R.37** — Don't pass a raw pointer/reference from an aliased smart pointer; take a local copy first.

## ES: Expressions and Statements

### General

- **ES.1** — Prefer the standard library to handcrafted code.
- **ES.2** — Prefer suitable abstractions to direct language features.
- **ES.3** — Don't repeat yourself; avoid redundant code.

### Declarations

- **ES.5** — Keep scopes small. Declare variables where used.
- **ES.6** — Declare names in for-statement initializers and conditions to limit scope.
- **ES.7** — Short names for local/common; longer for non-local/uncommon.
- **ES.8** — Avoid similar-looking names.
- **ES.9** — Avoid `ALL_CAPS` names (reserved for macros).
- **ES.10** — One name per declaration.
- **ES.11** — Use `auto` to avoid redundant type repetition.
- **ES.12** — Don't reuse names in nested scopes (shadowing).
- **ES.20** — Always initialize an object.
- **ES.21** — Don't introduce a variable before you need it.
- **ES.22** — Don't declare a variable until you have a value to initialize it with.
- **ES.23** — Prefer `{}`-initializer syntax (no narrowing, no parsing ambiguities).
- **ES.24** — Use `unique_ptr<T>` for owning pointers.
- **ES.25** — Declare objects `const`/`constexpr` unless modification is intended.
- **ES.26** — Don't recycle variables for unrelated purposes.
- **ES.27** — Use `std::array` or `stack_array` for stack arrays.
- **ES.28** — Use lambdas for complex initialization, especially of `const` variables.
- **ES.30** — Don't use macros for program text manipulation.
- **ES.31** — Don't use macros for constants or "functions".
- **ES.32** — Use `ALL_CAPS` for all macro names.
- **ES.33** — Give macros unique (prefixed) names.
- **ES.34** — Don't define C-style variadic functions.

### Expressions

- **ES.40** — Avoid complicated expressions.
- **ES.41** — Parenthesize when in doubt about precedence.
- **ES.42** — Keep pointer use simple; prefer `span` over pointer arithmetic.
- **ES.43** — Avoid expressions with undefined order of evaluation (e.g., `v[i] = ++i`).
- **ES.44** — Don't depend on order of evaluation of function arguments.
- **ES.45** — Use symbolic constants, not magic numbers.
- **ES.46** — Avoid narrowing conversions; use `gsl::narrow`/`gsl::narrow_cast`.
- **ES.47** — Use `nullptr`, not `0` or `NULL`.
- **ES.48** — Avoid casts.
- **ES.49** — If you must cast, use named casts (`static_cast`, `const_cast`, `dynamic_cast`, `reinterpret_cast`).
- **ES.50** — Don't cast away `const`.
- **ES.55** — Avoid the need for range checking (use range-for, spans).
- **ES.56** — Write `std::move()` only when explicitly moving to another scope. Never `return std::move(local)`.
- **ES.60** — Avoid `new`/`delete` outside resource management functions.
- **ES.61** — `delete[]` for arrays, `delete` for non-arrays.
- **ES.62** — Don't compare pointers into different arrays.
- **ES.63** — Don't slice; avoid copying derived objects into base objects.
- **ES.64** — Use `T{e}` notation for construction (not `T(e)` or `(T)e`).
- **ES.65** — Don't dereference invalid pointers.

### Statements

- **ES.70** — Prefer `switch` over `if` when comparing against constants.
- **ES.71** — Prefer range-`for` over `for` when iterating all elements.
- **ES.72** — Prefer `for` over `while` when there's an obvious loop variable.
- **ES.73** — Prefer `while` over `for` when there's no obvious loop variable.
- **ES.74** — Declare loop variable in the `for`-initializer.
- **ES.75** — Avoid `do`-statements.
- **ES.76** — Avoid `goto` (except for breaking out of nested loops).
- **ES.77** — Minimize `break`/`continue` in loops.
- **ES.78** — Don't rely on implicit fallthrough; use `[[fallthrough]]`.
- **ES.79** — Use `default` to handle common cases only.
- **ES.84** — Don't declare unnamed local variables (temporaries).
- **ES.85** — Make empty statements visible (use `{}` or a comment).
- **ES.86** — Avoid modifying loop control variables inside the body.
- **ES.87** — Don't add redundant `==`/`!=` to conditions (prefer `if (p)` over `if (p != nullptr)`).

### Arithmetic

- **ES.100** — Don't mix signed and unsigned arithmetic.
- **ES.101** — Use unsigned types for bit manipulation.
- **ES.102** — Use signed types for arithmetic.
- **ES.103** — Don't overflow.
- **ES.104** — Don't underflow.
- **ES.105** — Don't divide by integer zero.
- **ES.106** — Don't use `unsigned` to avoid negative values (it doesn't actually prevent them).
- **ES.107** — Don't use `unsigned` for subscripts; use `gsl::index`.

## Per: Performance

- **Per.1** — Don't optimize without reason.
- **Per.2** — Don't optimize prematurely.
- **Per.3** — Don't optimize non-performance-critical code.
- **Per.4** — Complicated code isn't necessarily faster.
- **Per.5** — Low-level code isn't necessarily faster.
- **Per.6** — Don't make performance claims without measurement.
- **Per.7** — Design to enable optimization (clean interfaces, compact data).
- **Per.10** — Rely on the static type system.
- **Per.11** — Move computation from run time to compile time (`constexpr`).
- **Per.12** — Eliminate redundant aliases.
- **Per.13** — Eliminate redundant indirections.
- **Per.14** — Minimize allocations/deallocations.
- **Per.15** — Don't allocate on critical branches.
- **Per.16** — Use compact data structures.
- **Per.17** — Declare the most-used member of a struct first.
- **Per.18** — Space is time (memory access dominates performance).
- **Per.19** — Access memory predictably (linear, contiguous).
- **Per.30** — Avoid context switches on the critical path.

## CP: Concurrency

### General

- **CP.1** — Assume code will run in a multi-threaded context.
- **CP.2** — Avoid data races (no concurrent access with at least one writer without synchronization).
- **CP.3** — Minimize explicit sharing of writable data.
- **CP.4** — Think in terms of tasks (`async`), not threads.
- **CP.8** — Don't use `volatile` for synchronization; use `atomic`.
- **CP.9** — Use tools (ThreadSanitizer, static analysis) to validate concurrent code.

### Concurrency

- **CP.20** — Use RAII for locks (`lock_guard`, `unique_lock`), never plain `lock()`/`unlock()`.
- **CP.21** — Use `std::lock()` or `scoped_lock` for multiple mutexes.
- **CP.22** — Never call unknown code (callbacks, virtual functions) while holding a lock.
- **CP.23** — A joining `thread` is a scoped container; prefer `gsl::joining_thread` or `std::jthread`.
- **CP.24** — A detached `thread` is a global container; only pass pointers to static/heap objects.
- **CP.25** — Prefer `gsl::joining_thread` over `std::thread`.
- **CP.26** — Don't `detach()` a thread.
- **CP.31** — Pass small data between threads by value.
- **CP.32** — Use `shared_ptr` for sharing ownership between unrelated threads.
- **CP.40** — Minimize context switching.
- **CP.41** — Minimize thread creation/destruction.
- **CP.42** — Don't `wait` without a condition (spurious wakeup).
- **CP.43** — Minimize time spent in critical sections.
- **CP.44** — Name your `lock_guard`s and `unique_lock`s (unnamed temporaries don't lock).
- **CP.50** — Define a `mutex` together with the data it guards; use `synchronized_value<T>`.

### Coroutines

- **CP.51** — Don't use capturing lambdas as coroutines (captures dangle after suspension).
- **CP.52** — Don't hold locks across suspension points.
- **CP.53** — Coroutine parameters should not be passed by reference.

### Message Passing

- **CP.60** — Use `future` to return values from concurrent tasks.
- **CP.61** — Use `async()` to spawn concurrent tasks.

### Lock-free

- **CP.100** — Don't use lock-free programming unless absolutely necessary.
- **CP.101** — Distrust hardware/compiler combinations; retest on changes.
- **CP.102** — Study the literature before shipping lock-free code.
- **CP.110** — Don't write your own double-checked locking; use `std::call_once` or C++11 static locals.
- **CP.111** — If you must double-check, use a conventional pattern with `atomic<bool>`.

### Other

- **CP.200** — `volatile` only for non-C++ memory (hardware registers, shared with other languages).

## E: Error Handling

- **E.1** — Develop error-handling strategy early.
- **E.2** — Throw exceptions to signal task failure.
- **E.3** — Use exceptions for error handling only (not normal control flow).
- **E.4** — Design error handling around invariants.
- **E.5** — Constructors establish invariants; throw if they can't.
- **E.6** — Use RAII to prevent leaks.
- **E.7** — State preconditions.
- **E.8** — State postconditions.
- **E.12** — Use `noexcept` when throwing is impossible or unacceptable.
- **E.13** — Never throw while being the direct owner of an object.
- **E.14** — Use user-defined exception types (not built-in types).
- **E.15** — Throw by value, catch by `const&`.
- **E.16** — Destructors, deallocation, `swap`, and exception copy/move must never fail.
- **E.17** — Don't try to catch every exception in every function.
- **E.18** — Minimize explicit `try`/`catch`; let RAII handle cleanup.
- **E.19** — Use `final_action` (`gsl::finally`) for cleanup when no suitable RAII handle exists.
- **E.25** — If no exceptions: simulate RAII with `valid()` checks.
- **E.26** — If no exceptions: consider failing fast (`abort()`).
- **E.27** — If no exceptions: use error codes systematically (return `pair<T, error>`).
- **E.28** — Avoid error handling based on global state (`errno`).
- **E.30** — Don't use exception specifications (deprecated).
- **E.31** — Order `catch`-clauses from most-derived to least-derived.

## Con: Constants and Immutability

- **Con.1** — By default, make objects immutable.
- **Con.2** — By default, make member functions `const`.
- **Con.3** — By default, pass pointers and references to `const`.
- **Con.4** — Use `const` for values that won't change after construction.
- **Con.5** — Use `constexpr` for values computable at compile time.

## T: Templates

### Use

- **T.1** — Use templates to raise abstraction level.
- **T.2** — Use templates to express algorithms over many types.
- **T.3** — Use templates for containers and ranges.
- **T.4** — Use templates for syntax tree manipulation.
- **T.5** — Combine generic and OO techniques to amplify strengths.

### Concepts (C++20)

- **T.10** — Specify concepts for all template arguments.
- **T.11** — Use standard concepts whenever possible.
- **T.12** — Prefer concept names over `auto` for local variables.
- **T.13** — Prefer shorthand (`sortable auto&`) for simple single-type concepts.
- **T.20** — Avoid concepts without meaningful semantics.
- **T.21** — Require a complete set of operations for a concept.
- **T.22** — Specify axioms for concepts (as comments).
- **T.23** — Differentiate refined concepts by adding new use patterns.
- **T.24** — Use tag classes/traits to differentiate concepts with same syntax but different semantics.
- **T.25** — Avoid complementary constraints (`C<T>` and `!C<T>`).
- **T.26** — Define concepts in terms of use-patterns, not just syntax.

### Interface

- **T.40** — Use function objects (lambdas) to pass operations to algorithms.
- **T.41** — Require only essential properties in concepts.
- **T.42** — Use template aliases to simplify notation.
- **T.43** — Prefer `using` over `typedef`.
- **T.44** — Use function templates to deduce class template args (`make_tuple`).
- **T.47** — Avoid highly visible unconstrained templates with common names.
- **T.48** — If no concepts: fake them with `enable_if`.
- **T.49** — Where possible, avoid type erasure.

### Definition

- **T.60** — Minimize a template's context dependencies.
- **T.61** — Don't over-parameterize members (SCARY: move non-dependent members to non-templated base).
- **T.62** — Place non-dependent members in a non-templated base class.
- **T.64** — Use specialization for alternative implementations of class templates.
- **T.65** — Use tag dispatch for alternative implementations of functions.
- **T.67** — Use specialization for irregular types.
- **T.68** — Use `{}` rather than `()` within templates to avoid ambiguities.
- **T.69** — Qualify non-member calls inside templates unless they're customization points.

### Templates and Hierarchies

- **T.80** — Don't naively templatize class hierarchies (code bloat from virtual functions).
- **T.81** — Don't mix hierarchies and arrays (slicing risk).
- **T.82** — Linearize a hierarchy when virtual functions are undesirable.
- **T.83** — Don't declare member function templates `virtual`.
- **T.84** — Use a non-template core for ABI-stable interfaces.

### Variadic Templates

- **T.100** — Use variadic templates for functions taking variable argument types.
- **T.103** — Don't use variadic templates for homogeneous argument lists (use `initializer_list`).

### Metaprogramming

- **T.120** — Use TMP only when really needed.
- **T.121** — Use TMP primarily to emulate concepts (pre-C++20).
- **T.122** — Use template aliases to compute types at compile time.
- **T.123** — Use `constexpr` functions to compute values at compile time.
- **T.124** — Prefer standard-library TMP facilities (`conditional`, `enable_if`, `tuple`).
- **T.125** — Use existing libraries for advanced TMP.

### Other

- **T.140** — Name reusable operations (see F.10).
- **T.141** — Use unnamed lambdas for one-off function objects (see F.11).
- **T.142** — Use template variables to simplify notation.
- **T.143** — Don't write unintentionally non-generic code.
- **T.144** — Don't specialize function templates; overload instead.
- **T.150** — Check class matches concept with `static_assert`.

## CPL: C-style Programming

- **CPL.1** — Prefer C++ to C.
- **CPL.2** — If you must use C, use the common C/C++ subset, compile as C++.
- **CPL.3** — Use C++ in calling code that interfaces with C.

## SF: Source Files

- **SF.1** — Use `.cpp` suffix for code, `.h` for interfaces (or project convention).
- **SF.2** — Headers must not contain object definitions or non-inline function definitions.
- **SF.3** — Use headers for all declarations used in multiple source files.
- **SF.4** — Include headers before other declarations in a file.
- **SF.5** — A `.cpp` file must include its own header first.
- **SF.6** — Use `using namespace` for transition, foundation libraries (`std`), or in local scopes only.
- **SF.7** — Don't write `using namespace` at global scope in headers.
- **SF.8** — Use `#include` guards (`#ifndef LIBRARY_FOOBAR_H`…).
- **SF.9** — Avoid cyclic dependencies among source files.
- **SF.10** — Avoid dependencies on implicitly `#include`d names; include what you use.
- **SF.11** — Headers should be self-contained.
- **SF.12** — Use `""` for relative includes, `<>` for system/external.
- **SF.13** — Use portable header identifiers in `#include` (forward slash, case-sensitive).
- **SF.20** — Use namespaces to express logical structure.
- **SF.21** — Don't use anonymous namespaces in headers.
- **SF.22** — Use anonymous namespaces for all internal/non-exported entities in `.cpp` files.

## SL: Standard Library

- **SL.1** — Use libraries wherever possible.
- **SL.2** — Prefer the standard library to other libraries.
- **SL.3** — Don't add non-standard entities to `namespace std`.
- **SL.4** — Use the standard library in a type-safe manner.

### Containers

- **SL.con.1** — Prefer `std::array` or `std::vector` over C arrays.
- **SL.con.2** — Use `std::vector` by default unless you have a specific reason.
- **SL.con.3** — Avoid bounds errors; use `at()` or `span`.
- **SL.con.4** — Don't use `memset`/`memcpy` on non-trivially-copyable types.

### String

- **SL.str.1** — Use `std::string` to own character sequences.
- **SL.str.2** — Use `std::string_view` or `span<char>` to refer to sequences.
- **SL.str.3** — Use `zstring`/`czstring` for C-style zero-terminated strings.
- **SL.str.4** — Use `char*` for a single character only.
- **SL.str.5** — Use `std::byte` for byte values not representing characters.
- **SL.str.10** — Use `std::string` when locale-sensitive operations are needed.
- **SL.str.11** — Use `gsl::span<char>` (not `string_view`) when mutation is needed.
- **SL.str.12** — Use the `s` suffix for `std::string` literals (`"hello"s`).

### Iostream

- **SL.io.1** — Use character-level input only when necessary.
- **SL.io.2** — Always consider ill-formed input when reading.
- **SL.io.3** — Prefer `iostream`s for I/O.
- **SL.io.10** — Call `ios_base::sync_with_stdio(false)` unless using `printf`.
- **SL.io.50** — Avoid `endl`; use `'\n'`.

## A: Architecture

- **A.1** — Separate stable code from less stable code.
- **A.2** — Express potentially reusable parts as a library.
- **A.4** — No cycles among libraries.

## Profiles (Subsets for Enforcement)

- **Type safety**: No `reinterpret_cast`, no C-style casts, no `const_cast`, no naked unions, no varargs, always initialize.
- **Bounds safety**: No pointer arithmetic, index only with constant expressions, no array-to-pointer decay, use bounds-checked interfaces.
- **Lifetime safety**: Don't dereference possibly-invalid pointers.

## GSL: Guidelines Support Library (Key Types)

| Type | Purpose |
|------|---------|
| `span<T>` | Non-owning view of contiguous range |
| `owner<T*>` | Mark raw owning pointer |
| `not_null<T>` | Pointer that must not be null |
| `zstring` / `czstring` | C-style string (nullable / const) |
| `Expects(cond)` | Precondition assertion |
| `Ensures(cond)` | Postcondition assertion |
| `finally(f)` | RAII cleanup action |
| `narrow<T>(x)` | Narrowing conversion that throws on loss |
| `narrow_cast<T>(x)` | Narrowing conversion (explicit) |
| `index` | Signed type for indexing (`ptrdiff_t`) |
| `joining_thread` | `std::thread` that joins in destructor |

## NL: Naming and Layout (Quick Defaults)

- **NL.1** — Don't repeat code in comments.
- **NL.2** — State intent in comments.
- **NL.3** — Keep comments crisp.
- **NL.4** — Consistent indentation.
- **NL.5** — Don't encode type in names.
- **NL.7** — Name length ∝ scope size.
- **NL.8** — Consistent naming style.
- **NL.9** — `ALL_CAPS` only for macros.
- **NL.10** — Prefer `underscore_style` names (default).
- **NL.11** — Make literals readable (digit separators: `1'000'000`, suffixes: `"hello"s`, `100ms`).
- **NL.15** — Use spaces sparingly.
- **NL.16** — Class member order: types → constructors → functions → data. `public` → `protected` → `private`.
- **NL.17** — K&R-derived layout (opening `{` on same line for classes, own line for functions).
- **NL.18** — C++-style declarator: `T& ref`, not `T &ref`.
- **NL.19** — Avoid easily misread names.
- **NL.20** — One statement per line.
- **NL.21** — One name per declaration.
- **NL.25** — Don't use `void` as argument type (`f()`, not `f(void)`).
- **NL.26** — Conventional `const` placement: `const int x`, not `int const x`.
- **NL.27** — `.cpp` for code, `.h` for headers (default).

