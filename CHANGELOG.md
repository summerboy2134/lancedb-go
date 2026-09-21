# Changelog

## [Unreleased]

### Changed
- **BREAKING (native segments)**: Native runtime upgraded from `lancedb 0.31.0` / Lance `8.0.0` to `lancedb 0.39.0` / Lance `12.0.0` (Arrow Rust `58.4.0`). The runtime identity string is now `lancedb=0.39.0;lance=12.0.0;arrow=58.4.0;rust=1.91.0;native-segment-wire=1`. Because model identity checksums embed the runtime version, **all model artifacts prepared against the 0.31.0/8.0.0 runtime are rejected** and must be re-prepared with `PrepareIndexModel`.
- The temporary, uncommitted `lance-index` vendor patch for SQ Dot distance is no longer needed and is not carried forward; Lance 12 ships the upstream fix (`fix: account for SQ offset in dot distance`, lancedb/lance#7481) plus widened u8 distance accumulators.
- The Rust build now always enables lancedb's `remote` feature. lancedb 0.38+ unconditionally compiles `src/job.rs`, which references the remote-gated `Error::Http` variant, so the published crate cannot build without the feature (upstream bug). Side effect: the native static library additionally links the remote client stack (tonic/arrow-flight/reqwest), so release binaries grow somewhat. The crate-level `remote` feature flag is kept as a no-op alias for compatibility.
- Notable upstream behavior changes to be aware of: datasets written by Lance 12 use the stable 2.2 file format and cannot be read by Lance 8 tooling (one-way upgrade for data files); `overwrite` no longer reuses fragment ids; HNSW construction is aligned with the reference algorithm (rebuild IvfHnsw* indexes to get corrected graphs); external manifest stores gain predecessor-conditioned publication, which hardens concurrent commits on S3-compatible stores.

### Added
- `ITable.VectorQuery(column string, vector []float32) IVectorQueryBuilder` — fluent builder for vector similarity searches, complementing the lower-level `VectorSearch` method.
- Input validation in `VectorQueryBuilder.Execute()`: returns clear errors for nil/empty vector, empty column name, and missing `Limit`.

### Removed
- **BREAKING**: `IVectorQueryBuilder.DistanceType(_ DistanceType) IVectorQueryBuilder` removed from the interface. The method was previously exported but documented as a no-op pending Rust FFI support.
- **BREAKING**: `DistanceType` type and constants (`DistanceTypeL2`, `DistanceTypeCosine`, `DistanceTypeDot`, `DistanceTypeHamming`) removed from `pkg/contracts`. Implementors of `IVectorQueryBuilder` and callers of `.DistanceType(...)` must remove those call sites.
- **BREAKING**: `IVectorQueryBuilder.Offset(offset int) IVectorQueryBuilder` removed from the interface. ANN vector search returns the K nearest neighbours and cannot be paginated with a row offset; exposing the method while always rejecting it at runtime was a public API lie. The underlying `VectorQueryBuilder` struct still validates and rejects non-zero offsets with a clear error.

### Fixed
- `ExecuteAsync` on both `QueryBuilder` and `VectorQueryBuilder` now always closes both returned channels after exactly one receives a value, satisfying Go's channel-close convention. Callers using `select` should use the two-value receive form (`value, ok := <-ch`) to distinguish a real value (`ok=true`) from a closed-empty channel (`ok=false`); on the closed-empty branch the other channel holds the actual result or error.
- Native segment commits now publish from their historical source snapshot instead of requiring it to remain the latest dataset version. The resulting Lance `Operation::CreateIndex` resolves intervening transactions using Lance's native conflict rules, so compatible appends succeed and leave their new fragments unindexed while source-fragment rewrites and other conflicts remain protected.
