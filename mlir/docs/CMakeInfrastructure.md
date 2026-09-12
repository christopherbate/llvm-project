# CMake Infrastructure

[TOC]

MLIR extends LLVM's CMake infrastructure with helpers for TableGen, libraries,
dialects, interfaces, tools, installation, and aggregate libraries. This guide
describes the MLIR-specific conventions implemented by
[`AddMLIR.cmake`](../cmake/modules/AddMLIR.cmake). The LLVM CMake documentation
still applies to the underlying LLVM helpers.

## Loading the MLIR CMake modules

The monorepo build loads the required modules. An out-of-tree project using an
installed MLIR package normally starts with:

~~~cmake
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
~~~

The project under `mlir/examples/standalone` is the canonical out-of-tree
template. It demonstrates package discovery, generated files, libraries,
tools, tests, and installation without depending on the monorepo build.

## Source and generated-file layout

Public declarations normally live below `mlir/include/mlir`, with
implementations in the corresponding directory below `mlir/lib`. For example,
a dialect declared in `include/mlir/Dialect/Foo/IR` is normally implemented by
a library in `lib/Dialect/Foo/IR`.

Generated targets such as `MLIRFooOpsIncGen` are build-local interface
libraries. The library that compiles or publishes those generated files lists
the target in its own `DEPENDS`. Other libraries normally depend on the logical
library through `LINK_LIBS`. A consumer that needs only a particular generated
header links its exact generation target instead.

Generated source files are different from published generated headers. A
source-generation target, such as one created for sharded operations, remains
an explicit `DEPENDS` entry of the library compiling those sources.

## TableGen

Set `LLVM_TARGET_DEFINITIONS`, call `mlir_tablegen` once for each output, and
finish with the helper matching the output:

| Helper | Intended output |
| --- | --- |
| `add_mlir_dialect` | Standard operation, type, and dialect fragments |
| `add_mlir_dialect_tablegen_target` | Other dialect-specific headers |
| `add_mlir_generic_tablegen_target` | Dialect-independent headers |
| `add_public_tablegen_target` | A library-specific header or source |

For example:

~~~cmake
set(LLVM_TARGET_DEFINITIONS FooPatterns.td)
mlir_tablegen(FooPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRFooPatternsIncGen)
~~~

The library that includes `FooPatterns.h.inc` then lists
`MLIRFooPatternsIncGen` in `DEPENDS`.

`mlir-generic-headers` collects dialect-independent public generation targets,
and `mlir-headers` also includes dialect-specific targets. They remain useful
as explicit compatibility and installation aggregates, but MLIR libraries do
not implicitly depend on either one. New code should model precise ownership
and library relationships instead.

### Dialects

The common dialect declaration is:

~~~cmake
add_mlir_dialect(FooOps foo)
~~~

This generates operation, type, and dialect declaration and definition
fragments and creates `MLIRFooOpsIncGen`. The implementation library lists that
target explicitly:

~~~cmake
add_mlir_dialect_library(MLIRFooDialect
  FooDialect.cpp
  FooOps.cpp

  DEPENDS
  MLIRFooOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
~~~

### Operation, type, and attribute interfaces

`add_mlir_interface(FooOpInterface)` emits operation-interface declaration and
definition fragments. `add_mlir_type_interface(FooTypeInterface)` does the same
for a type interface. Attribute interfaces and specialized interface forms use
the corresponding `mlir_tablegen` generators followed by a dialect or generic
TableGen target.

The library implementing an interface lists its generation target in
`DEPENDS`. A consumer links that interface library when its public or private
C++ interface uses the generated declarations.

### Passes

Pass declarations use `-gen-pass-decls`; C API fragments may be emitted from
the same `.td` file:

~~~cmake
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Foo)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Foo)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Foo)
add_mlir_dialect_tablegen_target(MLIRFooPassIncGen)
~~~

The library that defines or publishes these passes keeps
`MLIRFooPassIncGen` in `DEPENDS`. A consumer that includes only the generated
declarations can list `MLIRFooPassIncGen` directly in `LINK_LIBS` without
adding a link artifact.

### PDLL and generated documentation

Use `add_mlir_pdll_library` to compile a PDLL source and make its generated
output available to another target. Use `add_mlir_doc` for generated dialect,
operation, type, attribute, interface, or pass documentation. Documentation
targets are collected under the `mlir-doc` aggregate and are not compilation
prerequisites unless a source target explicitly consumes their output.

## Libraries

`add_mlir_library` is the base helper for MLIR libraries:

~~~cmake
add_mlir_library(MLIRFooTransforms
  FooTransforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Foo

  DEPENDS
  MLIRFooTransformsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRFooDialect
  MLIRPass
  )
~~~

`LINK_COMPONENTS` names LLVM components. `LINK_LIBS` names CMake or MLIR
library targets. Keeping them separate allows LLVM and MLIR to substitute their
monolithic shared libraries correctly.

Frequently used options include:

| Option | Purpose |
| --- | --- |
| `SHARED` or `OBJECT` | Select a non-default library form |
| `INSTALL_WITH_TOOLCHAIN` | Install with the toolchain distribution |
| `EXCLUDE_FROM_LIBMLIR` | Exclude the library from monolithic MLIR |
| `DISABLE_INSTALL` | Omit standard installation rules |
| `ENABLE_AGGREGATION` | Make objects available to an MLIR aggregate |
| `STANDALONE` | Do not add the implicit `LLVMSupport` dependency |
| `ADDITIONAL_HEADERS` | Associate individual headers with the target |
| `ADDITIONAL_HEADER_DIRS` | Add public headers to IDE source groups |
| `DEPENDS` | Add the library's non-library build prerequisites |

Prefer the wrapper describing a library's role:

| Helper | Additional behavior |
| --- | --- |
| `add_mlir_dialect_library` | Records a target in `MLIR_DIALECT_LIBS` |
| `add_mlir_conversion_library` | Records it in `MLIR_CONVERSION_LIBS` |
| `add_mlir_extension_library` | Records it in `MLIR_EXTENSION_LIBS` |
| `add_mlir_translation_library` | Records it in `MLIR_TRANSLATION_LIBS` |
| `add_mlir_example_library` | Applies the conventions to examples |
| `add_mlir_public_c_api_library` | Creates an aggregatable C API library |

The global categories support tools and aggregates that intentionally collect
an entire class of libraries. Ordinary libraries should list only their actual
dependencies.

### Link visibility

Choose `LINK_LIBS` visibility from the C++ interface:

| Declaration | Meaning |
| --- | --- |
| `PUBLIC A` | The target and its consumers use `A` |
| `PRIVATE A` | Only the target implementation uses `A` |
| `INTERFACE A` | Only consumers use `A` |

Unqualified entries retain CMake's legacy signature behavior. New code should
use explicit visibility when the distinction matters.

Use `mlir_target_link_libraries` when adding links after an MLIR target was
created, particularly for a library excluded from `libMLIR`. It applies
`MLIR_LINK_MLIR_DYLIB` substitution. Generated-header ordering follows the
resulting target graph automatically.

### Generated-header dependencies

Three rules cover generated headers:

1. A library lists its own TableGen and generated-source targets in `DEPENDS`.
2. `LINK_LIBS` orders compilation after generated headers in the exact,
   configuration-dependent compile interface evaluated by CMake. Private
   implementation requirements do not propagate to downstream consumers.
3. A consumer that needs a generated header without the provider's archive
   names the exact `*IncGen` interface target in `LINK_LIBS`.

TableGen targets have no link artifact, so they can use ordinary `LINK_LIBS`
with `PRIVATE`, `PUBLIC`, and `INTERFACE` visibility. Document the source-level
include that requires every direct generation dependency:

~~~cmake
# FooAnalysis.cpp includes the generated BarEnums.h.inc.
LINK_LIBS PRIVATE
MLIRBarEnumsIncGen
~~~

The LLVM library helper automatically keeps generation targets out of installed
and exported interfaces; installed packages already contain the generated
headers.

### How ordering is modeled

`add_public_tablegen_target` creates one build-local interface library. Private
sources make it a build target owning every generated output, while its
interface `HEADERS` file set publishes only the non-compilable outputs.
Generated C++ source shards therefore remain ordinary compilable sources. A
logical library publishes generated headers only for the marked targets in its
explicit `DEPENDS` list.

LLVM declares the generation targets as a CMake custom transitive compile
property. CMake 3.31 evaluates that property over the real link graph,
including aliases, forward references, visibility, configuration-dependent
generator expressions, and ordinary `target_link_libraries` calls. Entries
guarded by `LINK_ONLY` are excluded, as required for a compile interface.

For LLVM's split logical/object libraries, a phony target depends on the
evaluated generation-target list, and the compiling object target depends on
that phony target. This adds no provider libraries to the object-library graph,
so cyclic static-library relationships remain valid. Unlike a stamp file, the
phony target preserves the direct path to each generator in build-tool graph
analysis. Ordinary CMake targets consume the linked interface file sets
directly. `BUILD_LOCAL_INTERFACE` keeps generation targets out of installed
exports.

## C API libraries and aggregation

`add_mlir_public_c_api_library` creates an object-enabled MLIR library with the
visibility definitions needed by the C API. Use `add_mlir_aggregate` to build a
shared or static library from such components. `EMBED_LIBS` contribute their
objects; `PUBLIC_LIBS` remain normal exported link dependencies.

Aggregation metadata is exported only for libraries created with
`ENABLE_AGGREGATION`. `MLIR_INSTALL_AGGREGATE_OBJECTS` controls whether the
object libraries needed by an out-of-tree aggregate are installed. Imported
components must have been installed with compatible aggregation metadata.

## Tools, installation, and exports

Use `add_mlir_tool` for MLIR command-line tools. It delegates to LLVM's tool
infrastructure and participates in the normal runtime and installation layout.

`add_mlir_library` installs and exports its target by default.
`add_mlir_library_install` exposes the same rules for a non-standard library
construction path. `DISABLE_INSTALL` suppresses those rules, while
`INSTALL_WITH_TOOLCHAIN` includes the library when only the toolchain component
is installed.

Installed packages contain generated headers already, so imported targets do
not contribute build-tree generator prerequisites. Exported logical library
interfaces, rather than private `*IncGen` target names, are the contract for
standalone consumers.

## Validating dependency changes

Generated-header correctness must be tested from empty generated-header state.
An incremental build can leave files behind, and a broad aggregate can generate
a missing header incidentally before its consumer compiles.

Configure a fresh Ninja build and first build representative leaf libraries at
normal parallelism:

~~~shell
cmake -S llvm -B <build> -G Ninja <configuration options>
cmake --build <build> --target \
  MLIRArmNeonDialect MLIRLinalgDialect --parallel 32
~~~

Then build the broader graph to populate compiler dependency files. Only after
compilation succeeds, run:

~~~shell
ninja -C <build> -t missingdeps
~~~

`missingdeps` compares generated-file producers with include relationships in
compiler depfiles. Running it before compilation cannot discover those
includes. The clean leaf build and populated-depfile audit cover different
failure modes.

When changing the CMake infrastructure, also configure the focused CMake tests
with Ninja and Unix Makefiles, test the oldest supported CMake release, and
configure `mlir/examples/standalone` against an installed or build-tree MLIR
package. Inspect generated object-order rules to confirm they reach generated
header producers without passing through provider archives.

## Common mistakes

* Depending on every generator owned by `MLIRFooDialect` when only
  `MLIRFooEnumsIncGen` is required.
* Linking `MLIRFooDialect` when only one of its generated headers is required.
* Moving dialect-specific generators into `mlir-generic-headers` to hide a
  missing logical dependency.
* Repairing a leaf race with the global `mlir-headers` aggregate without first
  identifying the owning library.
* Running `missingdeps` before compiler depfiles have been populated.
* Validating only an incremental or broad aggregate build.
