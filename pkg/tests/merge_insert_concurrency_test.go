//go:build mergeinsert_concurrency

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

package tests

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	miniogo "github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	tcminio "github.com/testcontainers/testcontainers-go/modules/minio"

	"github.com/eozsahin1993/lancedb-go/pkg/contracts"
	"github.com/eozsahin1993/lancedb-go/pkg/lancedb"
)

const (
	concurrentMergeCount = 200
	concurrentTableName  = "merge_insert_concurrency"
	minioTestBucket      = "merge-insert-concurrency"
	minioTestRegion      = "us-east-1"
)

type mergeBackend struct {
	name    string
	uri     string
	options *contracts.ConnectionOptions
}

type mergeOutcome struct {
	id       int32
	version  uint64
	attempts uint32
	err      error
}

func TestConcurrentMergeInsertBackends(t *testing.T) {
	loadRootDotEnv(t)

	t.Run("local", func(t *testing.T) {
		runConcurrentMergeInsert(t, mergeBackend{
			name: "local",
			uri:  t.TempDir(),
		})
	})

	t.Run("minio", func(t *testing.T) {
		if envBool(t, "LANCEDB_TEST_SKIP_MINIO", false) {
			t.Skip("MinIO subtest disabled by LANCEDB_TEST_SKIP_MINIO")
		}
		runConcurrentMergeInsert(t, startMinIOBackend(t))
	})

	t.Run("ufile", func(t *testing.T) {
		backend, ok := ufileBackendFromEnv(t)
		if !ok {
			t.Skip("UFile credentials are not configured in .env")
		}
		runConcurrentMergeInsert(t, backend)
	})

	t.Run("ufile-conditional-writes", func(t *testing.T) {
		backend, ok := ufileBackendFromEnv(t)
		if !ok {
			t.Skip("UFile credentials are not configured in .env")
		}
		runUFileConditionalWriteProbe(t, backend)
	})
}

func TestBatchMergeInsertBackends(t *testing.T) {
	loadRootDotEnv(t)

	t.Run("local", func(t *testing.T) {
		runBatchMergeInsert(t, mergeBackend{
			name: "local",
			uri:  t.TempDir(),
		})
	})

	t.Run("minio", func(t *testing.T) {
		if envBool(t, "LANCEDB_TEST_SKIP_MINIO", false) {
			t.Skip("MinIO subtest disabled by LANCEDB_TEST_SKIP_MINIO")
		}
		runBatchMergeInsert(t, startMinIOBackend(t))
	})

	t.Run("ufile", func(t *testing.T) {
		backend, ok := ufileBackendFromEnv(t)
		if !ok {
			t.Skip("UFile credentials are not configured in .env")
		}
		runBatchMergeInsert(t, backend)
	})
}

func runBatchMergeInsert(t *testing.T, backend mergeBackend) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	conn, err := lancedb.Connect(ctx, backend.uri, backend.options)
	if err != nil {
		t.Fatalf("%s: connect: %v", backend.name, err)
	}

	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int32, Nullable: false},
		{Name: "writer", Type: arrow.PrimitiveTypes.Int32, Nullable: false},
	}, nil)
	schema, err := lancedb.NewSchema(arrowSchema)
	if err != nil {
		_ = conn.Close()
		t.Fatalf("%s: create schema: %v", backend.name, err)
	}

	created, err := conn.CreateTable(ctx, concurrentTableName, schema)
	if err != nil {
		_ = conn.Close()
		t.Fatalf("%s: create table: %v", backend.name, err)
	}
	if err := created.Close(); err != nil {
		_ = conn.Close()
		t.Fatalf("%s: close created table handle: %v", backend.name, err)
	}

	table, err := conn.OpenTable(ctx, concurrentTableName)
	if err != nil {
		_ = conn.Close()
		t.Fatalf("%s: open table: %v", backend.name, err)
	}
	defer table.Close()

	// Build a single Arrow record containing all 200 rows and do one
	// MergeInsert call. This avoids commit contention entirely.
	record := batchMergeRecords(arrowSchema, concurrentMergeCount)
	defer record.Release()

	mergeTimeout := envDuration(t, "LANCEDB_TEST_MERGE_TIMEOUT")
	builder := table.MergeInsert([]string{"id"}).
		WhenMatchedUpdateAll(nil).
		WhenNotMatchedInsertAll()
	if mergeTimeout > 0 {
		builder = builder.Timeout(mergeTimeout)
	}

	result, executeErr := builder.Execute(ctx, []arrow.Record{record})
	if err := conn.Close(); err != nil {
		t.Errorf("%s: close connection: %v", backend.name, err)
	}

	if executeErr != nil {
		t.Fatalf("%s: batch MergeInsert failed: %v", backend.name, executeErr)
	}

	t.Logf(
		"%s batch result: version=%d attempts=%d inserted=%d updated=%d deleted=%d",
		backend.name,
		result.Version,
		result.NumAttempts,
		result.NumInsertedRows,
		result.NumUpdatedRows,
		result.NumDeletedRows,
	)

	finalCount, missing, duplicates, unexpected := verifyConcurrentRows(t, ctx, backend)
	t.Logf(
		"%s batch verify: final_rows=%d missing=%d duplicates=%d unexpected=%d",
		backend.name,
		finalCount,
		len(missing),
		len(duplicates),
		len(unexpected),
	)
	if finalCount != concurrentMergeCount {
		t.Errorf("%s: final row count = %d, want %d", backend.name, finalCount, concurrentMergeCount)
	}
	if len(missing) > 0 {
		t.Errorf("%s: missing ids: %v", backend.name, missing)
	}
	if len(duplicates) > 0 {
		t.Errorf("%s: duplicate ids: %v", backend.name, duplicates)
	}
	if len(unexpected) > 0 {
		t.Errorf("%s: unexpected ids: %v", backend.name, unexpected)
	}
}

func batchMergeRecords(schema *arrow.Schema, count int32) arrow.Record {
	pool := memory.NewGoAllocator()

	idBuilder := array.NewInt32Builder(pool)
	defer idBuilder.Release()
	writerBuilder := array.NewInt32Builder(pool)
	defer writerBuilder.Release()

	for i := int32(0); i < count; i++ {
		idBuilder.Append(i)
		writerBuilder.Append(i)
	}

	idArray := idBuilder.NewArray()
	defer idArray.Release()
	writerArray := writerBuilder.NewArray()
	defer writerArray.Release()

	return array.NewRecord(schema, []arrow.Array{idArray, writerArray}, int64(count))
}

func runConcurrentMergeInsert(t *testing.T, backend mergeBackend) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Minute)
	defer cancel()

	conn, err := lancedb.Connect(ctx, backend.uri, backend.options)
	if err != nil {
		t.Fatalf("%s: connect: %v", backend.name, err)
	}

	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int32, Nullable: false},
		{Name: "writer", Type: arrow.PrimitiveTypes.Int32, Nullable: false},
	}, nil)
	schema, err := lancedb.NewSchema(arrowSchema)
	if err != nil {
		_ = conn.Close()
		t.Fatalf("%s: create schema: %v", backend.name, err)
	}

	created, err := conn.CreateTable(ctx, concurrentTableName, schema)
	if err != nil {
		_ = conn.Close()
		t.Fatalf("%s: create table: %v", backend.name, err)
	}
	if err := created.Close(); err != nil {
		_ = conn.Close()
		t.Fatalf("%s: close created table handle: %v", backend.name, err)
	}

	// Open every handle before releasing the writers. This intentionally gives
	// all workers the same initial dataset version and maximizes commit
	// contention without involving the project gRPC service.
	tables := make([]contracts.ITable, concurrentMergeCount)
	for i := range tables {
		tables[i], err = conn.OpenTable(ctx, concurrentTableName)
		if err != nil {
			closeTables(tables[:i])
			_ = conn.Close()
			t.Fatalf("%s: open worker table %d: %v", backend.name, i, err)
		}
	}

	mergeTimeout := envDuration(t, "LANCEDB_TEST_MERGE_TIMEOUT")
	start := make(chan struct{})
	outcomes := make(chan mergeOutcome, concurrentMergeCount)
	var workers sync.WaitGroup
	workers.Add(concurrentMergeCount)

	for i, table := range tables {
		go func(id int32, table contracts.ITable) {
			defer workers.Done()
			defer table.Close()

			record := singleMergeRecord(arrowSchema, id)
			defer record.Release()

			<-start
			builder := table.MergeInsert([]string{"id"}).
				WhenMatchedUpdateAll(nil).
				WhenNotMatchedInsertAll()
			if mergeTimeout > 0 {
				builder = builder.Timeout(mergeTimeout)
			}

			result, executeErr := builder.Execute(ctx, []arrow.Record{record})
			outcome := mergeOutcome{id: id, err: executeErr}
			if result != nil {
				outcome.version = result.Version
				outcome.attempts = result.NumAttempts
			}
			outcomes <- outcome
		}(int32(i), table)
	}

	close(start)
	workers.Wait()
	close(outcomes)

	if err := conn.Close(); err != nil {
		t.Errorf("%s: close writer connection: %v", backend.name, err)
	}

	var (
		executeErrors []mergeOutcome
		successes     int
		maxAttempts   uint32
		totalAttempts uint64
		maxVersion    uint64
	)
	for outcome := range outcomes {
		if outcome.err != nil {
			executeErrors = append(executeErrors, outcome)
			continue
		}
		successes++
		totalAttempts += uint64(outcome.attempts)
		if outcome.attempts > maxAttempts {
			maxAttempts = outcome.attempts
		}
		if outcome.version > maxVersion {
			maxVersion = outcome.version
		}
	}
	sort.Slice(executeErrors, func(i, j int) bool {
		return executeErrors[i].id < executeErrors[j].id
	})

	finalCount, missing, duplicates, unexpected := verifyConcurrentRows(t, ctx, backend)
	t.Logf(
		"%s result: successful=%d failed=%d final_rows=%d missing=%d duplicates=%d unexpected=%d max_attempts=%d max_version=%d",
		backend.name,
		successes,
		len(executeErrors),
		finalCount,
		len(missing),
		len(duplicates),
		len(unexpected),
		maxAttempts,
		maxVersion,
	)
	if successes > 0 {
		t.Logf("%s result: average_attempts=%.2f", backend.name, float64(totalAttempts)/float64(successes))
	}

	if len(executeErrors) > 0 {
		const maxReportedErrors = 20
		reported := executeErrors
		if len(reported) > maxReportedErrors {
			reported = reported[:maxReportedErrors]
		}
		var details strings.Builder
		for _, outcome := range reported {
			fmt.Fprintf(&details, "\n  id=%d: %v", outcome.id, outcome.err)
		}
		if len(executeErrors) > len(reported) {
			fmt.Fprintf(&details, "\n  ... %d additional errors omitted", len(executeErrors)-len(reported))
		}
		t.Errorf("%s: %d MergeInsert calls failed:%s", backend.name, len(executeErrors), details.String())
	}
	if finalCount != concurrentMergeCount {
		t.Errorf("%s: final row count = %d, want %d", backend.name, finalCount, concurrentMergeCount)
	}
	if len(missing) > 0 {
		t.Errorf("%s: missing ids: %v", backend.name, missing)
	}
	if len(duplicates) > 0 {
		t.Errorf("%s: duplicate ids: %v", backend.name, duplicates)
	}
	if len(unexpected) > 0 {
		t.Errorf("%s: unexpected ids: %v", backend.name, unexpected)
	}
}

func verifyConcurrentRows(
	t *testing.T,
	ctx context.Context,
	backend mergeBackend,
) (int64, []int32, []int32, []int32) {
	t.Helper()

	conn, err := lancedb.Connect(ctx, backend.uri, backend.options)
	if err != nil {
		t.Fatalf("%s: reconnect for verification: %v", backend.name, err)
	}
	defer conn.Close()
	defer func() {
		// Temporarily disabled cleanup to inspect bucket contents
		t.Logf("%s: skipping DropTable cleanup for inspection", backend.name)
	}()

	table, err := conn.OpenTable(ctx, concurrentTableName)
	if err != nil {
		t.Fatalf("%s: reopen table for verification: %v", backend.name, err)
	}
	defer table.Close()

	count, err := table.Count(ctx)
	if err != nil {
		t.Fatalf("%s: count final rows: %v", backend.name, err)
	}
	rows, err := table.Select(ctx, contracts.QueryConfig{Columns: []string{"id"}})
	if err != nil {
		t.Fatalf("%s: read final ids: %v", backend.name, err)
	}

	seen := make(map[int32]int, len(rows))
	var unexpected []int32
	for rowIndex, row := range rows {
		id, ok := rowID(row["id"])
		if !ok {
			t.Errorf("%s: row %d has an invalid id value of type %T", backend.name, rowIndex, row["id"])
			continue
		}
		seen[id]++
		if id < 0 || id >= concurrentMergeCount {
			unexpected = append(unexpected, id)
		}
	}

	var missing []int32
	var duplicates []int32
	for id := int32(0); id < concurrentMergeCount; id++ {
		switch {
		case seen[id] == 0:
			missing = append(missing, id)
		case seen[id] > 1:
			duplicates = append(duplicates, id)
		}
	}
	sort.Slice(unexpected, func(i, j int) bool { return unexpected[i] < unexpected[j] })
	return count, missing, duplicates, unexpected
}

func singleMergeRecord(schema *arrow.Schema, id int32) arrow.Record {
	pool := memory.NewGoAllocator()

	idBuilder := array.NewInt32Builder(pool)
	idBuilder.Append(id)
	idArray := idBuilder.NewArray()
	idBuilder.Release()
	defer idArray.Release()

	writerBuilder := array.NewInt32Builder(pool)
	writerBuilder.Append(id)
	writerArray := writerBuilder.NewArray()
	writerBuilder.Release()
	defer writerArray.Release()

	return array.NewRecord(schema, []arrow.Array{idArray, writerArray}, 1)
}

func closeTables(tables []contracts.ITable) {
	for _, table := range tables {
		if table != nil {
			_ = table.Close()
		}
	}
}

func rowID(value any) (int32, bool) {
	switch value := value.(type) {
	case float64:
		return int32(value), value == float64(int32(value))
	case int:
		return int32(value), int64(value) == int64(int32(value))
	case int32:
		return value, true
	case int64:
		return int32(value), value == int64(int32(value))
	case json.Number:
		parsed, err := value.Int64()
		return int32(parsed), err == nil && parsed == int64(int32(parsed))
	default:
		return 0, false
	}
}

func startMinIOBackend(t *testing.T) mergeBackend {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	container, err := tcminio.Run(
		ctx,
		"minio/minio:latest",
		tcminio.WithUsername("minioadmin"),
		tcminio.WithPassword("minioadmin"),
	)
	if err != nil {
		t.Fatalf("start MinIO container: %v", err)
	}
	t.Cleanup(func() {
		if err := container.Terminate(context.Background()); err != nil {
			t.Errorf("terminate MinIO container: %v", err)
		}
	})

	connectionString, err := container.ConnectionString(ctx)
	if err != nil {
		t.Fatalf("get MinIO connection string: %v", err)
	}
	client, err := miniogo.New(connectionString, &miniogo.Options{
		Creds:  credentials.NewStaticV4(container.Username, container.Password, ""),
		Secure: false,
	})
	if err != nil {
		t.Fatalf("create MinIO client: %v", err)
	}
	if err := client.MakeBucket(ctx, minioTestBucket, miniogo.MakeBucketOptions{Region: minioTestRegion}); err != nil {
		t.Fatalf("create MinIO bucket: %v", err)
	}

	return mergeBackend{
		name: "minio",
		uri:  uniqueDatabaseURI(fmt.Sprintf("s3://%s/sdk-test", minioTestBucket)),
		options: &contracts.ConnectionOptions{StorageOptions: map[string]string{
			contracts.StorageAccessKeyID:               container.Username,
			contracts.StorageSecretAccessKey:           container.Password,
			contracts.StorageEndpoint:                  "http://" + connectionString,
			contracts.StorageRegion:                    minioTestRegion,
			contracts.StorageAllowHTTP:                 "true",
			contracts.StorageVirtualHostedStyleRequest: "false",
		}},
	}
}

func ufileBackendFromEnv(t *testing.T) (mergeBackend, bool) {
	t.Helper()

	required := []string{
		"LANCEDB_TEST_UFILE_URI",
		"LANCEDB_TEST_UFILE_ENDPOINT",
		"LANCEDB_TEST_UFILE_ACCESS_KEY",
		"LANCEDB_TEST_UFILE_SECRET_KEY",
	}
	configured := false
	var missing []string
	for _, key := range required {
		if strings.TrimSpace(os.Getenv(key)) != "" {
			configured = true
		} else {
			missing = append(missing, key)
		}
	}
	if !configured {
		return mergeBackend{}, false
	}
	if len(missing) > 0 {
		t.Fatalf("incomplete UFile configuration; missing: %s", strings.Join(missing, ", "))
	}

	storageOptions := map[string]string{
		contracts.StorageAccessKeyID:     strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_ACCESS_KEY")),
		contracts.StorageSecretAccessKey: strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_SECRET_KEY")),
		contracts.StorageEndpoint:        strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_ENDPOINT")),
	}
	addStorageOption(storageOptions, contracts.StorageRegion, "LANCEDB_TEST_UFILE_REGION")
	addStorageOption(storageOptions, contracts.StorageSessionToken, "LANCEDB_TEST_UFILE_SESSION_TOKEN")
	addStorageOption(storageOptions, contracts.StorageAllowHTTP, "LANCEDB_TEST_UFILE_ALLOW_HTTP")
	addStorageOption(
		storageOptions,
		contracts.StorageVirtualHostedStyleRequest,
		"LANCEDB_TEST_UFILE_VIRTUAL_HOSTED_STYLE_REQUEST",
	)
	addStorageOption(storageOptions, contracts.StorageUnsignedPayload, "LANCEDB_TEST_UFILE_UNSIGNED_PAYLOAD")
	addStorageOption(storageOptions, contracts.StorageConditionalPut, "LANCEDB_TEST_UFILE_CONDITIONAL_PUT")
	addStorageOption(storageOptions, contracts.StorageCopyIfNotExists, "LANCEDB_TEST_UFILE_COPY_IF_NOT_EXISTS")
	addStorageOption(storageOptions, contracts.StorageDisableTagging, "LANCEDB_TEST_UFILE_DISABLE_TAGGING")

	return mergeBackend{
		name: "ufile",
		uri:  uniqueDatabaseURI(strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_URI"))),
		options: &contracts.ConnectionOptions{
			StorageOptions: storageOptions,
		},
	}, true
}

func runUFileConditionalWriteProbe(t *testing.T, backend mergeBackend) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	client, bucket, objectPrefix := newUFileS3Client(t, backend)
	createKey := objectPrefix + "/conditional-create"
	updateKey := objectPrefix + "/conditional-update"
	t.Cleanup(func() {
		for _, key := range []string{createKey, updateKey} {
			if err := client.RemoveObject(context.Background(), bucket, key, miniogo.RemoveObjectOptions{}); err != nil {
				t.Errorf("ufile conditional probe: remove %s: %v", filepath.Base(key), err)
			}
		}
	})

	const writers = 32
	createResults := make(chan error, writers)
	createStart := make(chan struct{})
	var createWorkers sync.WaitGroup
	createWorkers.Add(writers)
	for writer := 0; writer < writers; writer++ {
		go func(writer int) {
			defer createWorkers.Done()
			payload := []byte(fmt.Sprintf("create-writer-%d", writer))
			options := miniogo.PutObjectOptions{ContentType: "application/octet-stream"}
			options.SetMatchETagExcept("*")
			<-createStart
			_, err := client.PutObject(
				ctx,
				bucket,
				createKey,
				bytes.NewReader(payload),
				int64(len(payload)),
				options,
			)
			createResults <- err
		}(writer)
	}
	close(createStart)
	createWorkers.Wait()
	close(createResults)
	assertExactlyOneConditionalWrite(t, "If-None-Match create", createResults)

	seed := []byte("etag-seed")
	seedInfo, err := client.PutObject(
		ctx,
		bucket,
		updateKey,
		bytes.NewReader(seed),
		int64(len(seed)),
		miniogo.PutObjectOptions{ContentType: "application/octet-stream"},
	)
	if err != nil {
		t.Fatalf("ufile conditional probe: seed ETag object: %v", err)
	}
	if seedInfo.ETag == "" {
		t.Fatal("ufile conditional probe: seed object response did not include an ETag")
	}

	updateResults := make(chan error, writers)
	updateStart := make(chan struct{})
	var updateWorkers sync.WaitGroup
	updateWorkers.Add(writers)
	for writer := 0; writer < writers; writer++ {
		go func(writer int) {
			defer updateWorkers.Done()
			payload := []byte(fmt.Sprintf("update-writer-%d", writer))
			options := miniogo.PutObjectOptions{ContentType: "application/octet-stream"}
			options.SetMatchETag(seedInfo.ETag)
			<-updateStart
			_, err := client.PutObject(
				ctx,
				bucket,
				updateKey,
				bytes.NewReader(payload),
				int64(len(payload)),
				options,
			)
			updateResults <- err
		}(writer)
	}
	close(updateStart)
	updateWorkers.Wait()
	close(updateResults)
	assertExactlyOneConditionalWrite(t, "If-Match update", updateResults)
}

func newUFileS3Client(
	t *testing.T,
	backend mergeBackend,
) (*miniogo.Client, string, string) {
	t.Helper()

	databaseURI, err := url.Parse(backend.uri)
	if err != nil {
		t.Fatalf("parse LANCEDB_TEST_UFILE_URI: %v", err)
	}
	if databaseURI.Scheme != "s3" || databaseURI.Host == "" {
		t.Fatalf("LANCEDB_TEST_UFILE_URI must have the form s3://bucket/prefix")
	}

	endpointValue := strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_ENDPOINT"))
	endpointURL, err := url.Parse(endpointValue)
	if err != nil {
		t.Fatalf("parse LANCEDB_TEST_UFILE_ENDPOINT: %v", err)
	}
	if endpointURL.Host == "" || (endpointURL.Scheme != "http" && endpointURL.Scheme != "https") {
		t.Fatalf("LANCEDB_TEST_UFILE_ENDPOINT must be an http:// or https:// URL")
	}
	if endpointURL.Path != "" && endpointURL.Path != "/" {
		t.Fatalf("LANCEDB_TEST_UFILE_ENDPOINT must not contain a path")
	}

	bucketLookup := miniogo.BucketLookupAuto
	if value := strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_VIRTUAL_HOSTED_STYLE_REQUEST")); value != "" {
		virtualHosted, parseErr := strconv.ParseBool(value)
		if parseErr != nil {
			t.Fatalf("LANCEDB_TEST_UFILE_VIRTUAL_HOSTED_STYLE_REQUEST must be a boolean")
		}
		if virtualHosted {
			bucketLookup = miniogo.BucketLookupDNS
		} else {
			bucketLookup = miniogo.BucketLookupPath
		}
	}

	client, err := miniogo.New(endpointURL.Host, &miniogo.Options{
		Creds: credentials.NewStaticV4(
			strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_ACCESS_KEY")),
			strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_SECRET_KEY")),
			strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_SESSION_TOKEN")),
		),
		Secure:       endpointURL.Scheme == "https",
		Region:       strings.TrimSpace(os.Getenv("LANCEDB_TEST_UFILE_REGION")),
		BucketLookup: bucketLookup,
	})
	if err != nil {
		t.Fatalf("create UFile S3 client: %v", err)
	}

	objectPrefix := strings.Trim(databaseURI.Path, "/")
	if objectPrefix == "" {
		objectPrefix = "lancedb-go-conditional-probe"
	}
	return client, databaseURI.Host, objectPrefix
}

func assertExactlyOneConditionalWrite(t *testing.T, operation string, results <-chan error) {
	t.Helper()

	successes := 0
	preconditionFailures := 0
	var unexpected []string
	for err := range results {
		if err == nil {
			successes++
			continue
		}
		response := miniogo.ToErrorResponse(err)
		if response.StatusCode == http.StatusPreconditionFailed {
			preconditionFailures++
			continue
		}
		unexpected = append(unexpected, fmt.Sprintf("status=%d code=%q error=%v", response.StatusCode, response.Code, err))
	}

	t.Logf(
		"ufile conditional probe %s: successful=%d precondition_failed=%d unexpected=%d",
		operation,
		successes,
		preconditionFailures,
		len(unexpected),
	)
	if successes != 1 {
		t.Errorf("ufile conditional probe %s: successful writes = %d, want exactly 1", operation, successes)
	}
	if preconditionFailures != 31 {
		t.Errorf(
			"ufile conditional probe %s: HTTP 412 responses = %d, want 31",
			operation,
			preconditionFailures,
		)
	}
	if len(unexpected) > 0 {
		const maxReported = 10
		if len(unexpected) > maxReported {
			unexpected = unexpected[:maxReported]
		}
		t.Errorf("ufile conditional probe %s returned unexpected errors: %s", operation, strings.Join(unexpected, "; "))
	}
}

func addStorageOption(options map[string]string, storageKey, environmentKey string) {
	if value := strings.TrimSpace(os.Getenv(environmentKey)); value != "" {
		options[storageKey] = value
	}
}

func uniqueDatabaseURI(base string) string {
	return fmt.Sprintf(
		"%s/merge-insert-%d",
		strings.TrimRight(strings.TrimSpace(base), "/"),
		time.Now().UTC().UnixNano(),
	)
}

func envBool(t *testing.T, key string, fallback bool) bool {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		t.Fatalf("%s must be a boolean, got %q", key, value)
	}
	return parsed
}

func envDuration(t *testing.T, key string) time.Duration {
	t.Helper()
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return 0
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		t.Fatalf("%s must be a Go duration such as 30s or 2m, got %q", key, value)
	}
	if parsed < 0 {
		t.Fatalf("%s must not be negative", key)
	}
	return parsed
}

func loadRootDotEnv(t *testing.T) {
	t.Helper()

	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate concurrency test source file")
	}
	dotEnvPath := filepath.Clean(filepath.Join(filepath.Dir(sourceFile), "..", "..", ".env"))
	file, err := os.Open(dotEnvPath)
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		t.Fatalf("open local .env: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for lineNumber := 1; scanner.Scan(); lineNumber++ {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		key, value, found := strings.Cut(line, "=")
		if !found {
			t.Fatalf(".env line %d must be KEY=VALUE", lineNumber)
		}
		key = strings.TrimSpace(key)
		if !validEnvironmentKey(key) {
			t.Fatalf(".env line %d has invalid key %q", lineNumber, key)
		}
		if _, alreadySet := os.LookupEnv(key); alreadySet {
			continue
		}
		parsedValue, err := parseDotEnvValue(strings.TrimSpace(value))
		if err != nil {
			t.Fatalf(".env line %d has invalid value for %s: %v", lineNumber, key, err)
		}
		t.Setenv(key, parsedValue)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("read local .env: %v", err)
	}
}

func validEnvironmentKey(key string) bool {
	for index, char := range key {
		if index == 0 {
			if char != '_' && !unicode.IsLetter(char) {
				return false
			}
			continue
		}
		if char != '_' && !unicode.IsLetter(char) && !unicode.IsDigit(char) {
			return false
		}
	}
	return key != ""
}

func parseDotEnvValue(value string) (string, error) {
	if value == "" {
		return "", nil
	}
	if strings.HasPrefix(value, `"`) {
		if !strings.HasSuffix(value, `"`) {
			return "", fmt.Errorf("unterminated double-quoted value")
		}
		return strconv.Unquote(value)
	}
	if strings.HasPrefix(value, "'") {
		if !strings.HasSuffix(value, "'") {
			return "", fmt.Errorf("unterminated single-quoted value")
		}
		return strings.TrimSuffix(strings.TrimPrefix(value, "'"), "'"), nil
	}
	return value, nil
}
