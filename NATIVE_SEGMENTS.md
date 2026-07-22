# Native Index Segment FFI

This fork exposes Lance's fragment-scoped distributed vector-index workflow
through a versioned Go/JSON/C/Rust boundary.  The implementation baseline is:

- `lancedb =0.31.0`
- Lance crates `=8.0.0`
- Arrow Rust crates `=58.0.0`
- Rust `1.91.0`
- native segment wire version `1`

The complete runtime identity is embedded in every response and model identity:

```text
lancedb=0.31.0;lance=8.0.0;arrow=58.0.0;rust=1.91.0;native-segment-wire=1
```

## Public Go capability

Native tables implement the optional `contracts.ITableNativeSegments` interface:

- `ListFragments(ctx, datasetVersion)`
- `PrepareIndexModel(ctx, request)`
- `CreateIndexUncommitted(ctx, request)`
- `MergeExistingIndexSegments(ctx, request)`
- `CommitExistingIndexSegments(ctx, request)`
- `InspectIndexSegments(ctx, indexName)`

Keeping this capability out of the existing `contracts.ITable` interface avoids
breaking downstream mocks and alternate table implementations.

Every training, build, physical merge, and commit request explicitly carries the fixed
dataset version, complete fragment coverage, vector column, logical index name,
index configuration and configuration digest.  Build and physical-merge requests
also carry an explicit model identity. `PrepareIndexModel` opens exactly the
requested dataset version, validates `training_fragment_ids` against that
snapshot, and passes that fragment set to Lance 8's public `build_ivf_model` and
`build_pq_model_in_fragments` APIs. It never trains from an implicit latest
snapshot. IVF_FLAT returns centroids; IVF_PQ returns centroids and the residual
PQ codebook trained from the same fragment range.

The caller supplies `model_id` and `model_scope`, but not a checksum. Rust
computes the canonical SHA-256 identity over the runtime, index config/digest,
model ID/scope, and raw centroid/codebook bytes. `CreateIndexUncommitted`
reloads the artifacts and recomputes that checksum before building, so an
arbitrary Go checksum or a changed referenced object is rejected.

`Float32Artifact` is a shaped `float32_le` byte tensor with exactly one storage
form. Small test models may use inline bytes; Go's JSON encoder and the Rust
envelope encode those bytes as standard base64. The default combined inline
limit is 1 MiB and callers can lower it with `inline_artifact_max_bytes`. IVF
centroid shape is
`[num_partitions, dimension]`; PQ codebook shape is
`[num_sub_vectors, 2^num_bits, dimension/num_sub_vectors]`.

Production models, and all models with 4096 or more dimensions, must set
`artifact_output_uri` to a local directory, `file://` directory, or shared
object-store prefix such as `s3://`, `gs://`, or `az://`. Rust writes each raw
tensor to a unique temporary local file/object and then renames/publishes it to
an immutable checksum-qualified name. The response contains a
`ModelArtifactReference` with URI, dtype, shape, byte length, checksum, and
runtime version instead of base64 data. A Worker passes that reference unchanged
to `CreateIndexUncommitted`; Rust loads it directly, verifies all metadata and
the raw-byte checksum, and gives Lance the reconstructed model. Cloud URI
support follows the corresponding native build feature (`aws`, `gcs`, or
`azure`) and credentials come from the Worker environment.

Dimensions of 4096 or greater require `artifact_output_uri` even when a highly
reduced test configuration would happen to fit below the byte limit. This keeps
the production contract from silently falling back to repeated JSON/base64
copies for high-dimensional models.

One Macro build should prepare a single model, distribute that identity and
artifact reference to its Micro Groups, and merge only the resulting compatible
segments. Independently trained models retain different identities and are not
eligible for physical merge even if their configurations match.

## Versioned metadata envelope

Lance's Rust `IndexMetadata` is not a Go ABI.  Wire version 1 exposes only stable
fields needed for orchestration:

- segment UUID and logical index name;
- source dataset version and fragment bitmap;
- vector column, index version and index details;
- index config/digest and model identity for uncommitted segments;
- the complete protobuf-encoded `IndexMetadata` as opaque bytes.

Rust decodes the opaque protobuf before merge or commit and verifies every stable
field against it.  This prevents a caller from changing UUID, fragment coverage,
details or version while retaining unrelated opaque metadata.

Coverage must be disjoint and complete.  A physical merge additionally requires
identical config, config digest and model identity.  A logical commit may contain
multiple physical segments with different model identities; those segments are
committed together without pretending that their physical models are mergeable.
Commit also requires the requested version to still be the table's current
version and requires exact coverage of every fragment in that snapshot.

Same-name rebuilds are staged with Lance's `replace(true)` builder option.
`execute_uncommitted` only writes the new physical segment and does not change
the manifest, so inspect and query continue to see the previously committed
generation throughout prepare/build/merge. `commit_existing_index_segments`
then atomically replaces the logical index with the new segment UUIDs. If an
append or other write advances the dataset version first, commit rejects the
stale generation and leaves the old logical index unchanged.

## C ownership and failures

All Native Segment requests are length-delimited UTF-8 JSON bytes.  Responses are
length-delimited JSON bytes allocated by Rust and released with
`simple_lancedb_free_bytes`.  C strings use `simple_lancedb_free_string`, and
`SimpleResult` uses `simple_lancedb_result_free`.

Every new FFI entry point is protected by `catch_unwind`.  `SimpleResult` carries
an error code (`0` success, `1` operation error, `2` converted Rust panic) and the
runtime version, in addition to the legacy success and error-message fields.

## Actual 0.24 to 0.31 adjustments

- `Scannable` write inputs now use a `RecordBatch`/`Vec<RecordBatch>` or a boxed
  `RecordBatchReader`, depending on the call site.
- removed the obsolete index-statistics `loss` field.
- preserved the existing Go nullability-widening contract: although 0.31 can
  tighten a currently null-free nullable column, the FFI still rejects that
  operation as older releases did.
- distributed builds use `DatasetIndexExt::create_index_builder(...).fragments(...)`
  followed by `execute_uncommitted`; no `IndexSegmentBuilder` is used.
- physical merge uses `merge_existing_index_segments` and publication uses
  `commit_existing_index_segments` on the same Lance 8 dataset runtime.
- precomputed models are installed through `IvfBuildParams::try_with_centroids`
  and `PQBuildParams::with_codebook`.
- shared models are trained through Lance 8's public `build_ivf_model` and
  `build_pq_model_in_fragments` functions with explicit fragment IDs.
- opaque metadata uses Lance 8's `lance_table::format::pb::IndexMetadata`
  protobuf conversion instead of exposing the unstable Rust struct layout.
- `Table::dataset()` / `DatasetConsistencyWrapper` is updated after commit so
  subsequent query and write operations observe the committed manifest version.
- `rust/.cargo/config.toml` uses Cargo's incompatible-Rust-version fallback and
  `Cargo.lock` pins the AWS/Smithy patch releases that still support Rust 1.91.0;
  newer patches in the same semver ranges already require Rust 1.91.1.

`rust/build.rs` is the single source of header generation and reads
`rust/cbindgen.toml`.  `scripts/build-native.sh` copies that exact generated header
and uses ordinary Cargo for native builds; Zig is only needed for cross builds.
Lance 8 also requires `protoc`; the script accepts a pinned `PROTOC` and optional
`PROTOC_INCLUDE`, or discovers `protoc` on `PATH`.
