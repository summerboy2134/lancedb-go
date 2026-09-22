# Direct SDK MergeInsert concurrency test

This test bypasses project gRPC and proxy services. It calls `lancedb-go`
directly through CGO and compares 200 concurrent `MergeInsert` transactions
against:

1. the local filesystem;
2. an automatically managed MinIO container;
3. UFile's S3-compatible endpoint, when local credentials are configured.

The UFile run also sends two direct S3 conditional-write probes:

- 32 concurrent `If-None-Match: *` creates of one object;
- 32 concurrent `If-Match: <same-old-etag>` updates of one object.

Exactly one request must succeed in each probe and the other 31 requests must
return HTTP 412. This isolates the object-store primitive from Lance's commit
implementation.

## Configure UFile

Fill in the repository-root `.env` file. It is ignored by Git. At minimum:

```dotenv
LANCEDB_TEST_UFILE_URI=s3://your-test-bucket/disposable-prefix
LANCEDB_TEST_UFILE_ENDPOINT=https://your-ufile-s3-endpoint
LANCEDB_TEST_UFILE_ACCESS_KEY=your-public-key
LANCEDB_TEST_UFILE_SECRET_KEY=your-private-key
```

Use a dedicated test bucket or disposable prefix. The test creates a unique
database prefix on every run and drops its table afterward.

Optional object-store settings are documented in `.env.example`. Environment
variables override values loaded from `.env`.

## Run

Docker must be available for the MinIO comparison:

```bash
make test-merge-insert-concurrency
```

To run only one backend:

```bash
CGO_CFLAGS="-I$(pwd)/include" \
CGO_LDFLAGS="$(pwd)/lib/darwin_arm64/liblancedb_go.a -framework Security -framework CoreFoundation -framework SystemConfiguration" \
go test -v -tags mergeinsert_concurrency -timeout 15m \
  ./pkg/tests -run '^TestConcurrentMergeInsertBackends$/local$' -count=1
```

Adjust the native library path and linker flags for non-Apple-Silicon
platforms, or use the values shown by `make platform-info`.

## Interpretation

- Only UFile loses rows: investigate UFile conditional write / atomic commit
  compatibility and its `conditional_put` / `copy_if_not_exists` settings.
- Every backend reports contention errors: investigate retry limits and
  conflict handling in Lance/lancedb-go.
- Calls report success but final IDs are missing: treat this as a silent atomic
  commit correctness failure, not an ordinary retry exhaustion.

## Findings (2026-09-21, lance 12.0.0 / lancedb 0.39.0 runtime)

- local: `successful=200 failed=0 missing=0 duplicates=0 max_attempts=1` — PASS.
- ufile merge probe: `successful=200 failed=0 final_rows=1 missing=199
  max_attempts=1 max_version=2` — every call reported success without a single
  retry, yet 199 transactions silently vanished (last-writer-wins).
- ufile raw conditional-write probe: **the UFile endpoint ignores S3
  conditional-write preconditions entirely.**
  - 32 concurrent `If-None-Match: *` creates: 0 HTTP 412 responses (want 31).
  - 32 concurrent `If-Match: <same-old-etag>` updates: all 32 succeeded
    (want exactly 1).

Conclusion: the row loss is a UFile server-side defect, not a Lance commit
logic bug. Lance's S3 commit protocol relies on conditional writes for atomic
publication; a store that silently drops `If-None-Match`/`If-Match` turns every
concurrent commit into an undetected overwrite. Lance 12's predecessor-
conditioned publication cannot compensate because it is built on the same
primitive. Until UFile implements conditional writes (AWS S3 has supported
`If-None-Match` since 2024; MinIO supports both), concurrent writers against
UFile must be serialized at the application level, or commits must be routed
through an external lock (e.g. Lance's `dynamodb_table` commit handler).

Follow-up verification with `LANCEDB_TEST_UFILE_CONDITIONAL_PUT=disabled`:
every commit — starting from `create table` — now fails loudly with
`Operation put_opts with mode PutMode::Create when conditional put is disabled
not yet implemented by AmazonS3`. This confirms the safety net: the option
converts silent corruption into hard failure (UFile becomes write-unusable
through the SDK), and it confirms that Lance 8/12 always commit `s3://`
datasets through `ConditionalPutCommitHandler`, so `copy_if_not_exists` never
applies to the S3 commit path (it only serves the rename-based handler used
for `file://` on Windows).

Production exposure note: the production write path (lancedb-prj gRPC service)
does not rely on UFile conditional writes. It serializes all mutations per
`(bucket, index)` through a bounded FIFO single-writer queue
(`internal/store/collection_mutation.go`), buffers inserts in a WAL
(`internal/store/wal.go`), and coordinates shard ownership via ZooKeeper
(`internal/cluster/zk.go`). This probe intentionally bypasses that service,
so its UFile failure demonstrates the raw-storage hazard for any direct-SDK
writer, not a production regression. Direct-SDK access to UFile-backed
datasets must remain serialized (or use `conditional_put=disabled` as a
fuse), and any future bypass tooling (migrations, backups, manual SDK use)
inherits the same constraint.
