// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

package tests

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/stretchr/testify/require"

	"github.com/eozsahin1993/lancedb-go/pkg/contracts"
	"github.com/eozsahin1993/lancedb-go/pkg/internal"
	"github.com/eozsahin1993/lancedb-go/pkg/lancedb"
)

const nativeSegmentDimension = 4

func appendNativeSegmentRows(
	t *testing.T,
	ctx context.Context,
	table contracts.ITable,
	schema *arrow.Schema,
	start int32,
	count int,
) {
	t.Helper()
	pool := memory.NewGoAllocator()
	idBuilder := array.NewInt32Builder(pool)
	vectorBuilder := array.NewFixedSizeListBuilder(pool, nativeSegmentDimension, arrow.PrimitiveTypes.Float32)
	valuesBuilder := vectorBuilder.ValueBuilder().(*array.Float32Builder)
	defer idBuilder.Release()
	defer vectorBuilder.Release()

	for offset := 0; offset < count; offset++ {
		id := start + int32(offset)
		idBuilder.Append(id)
		vectorBuilder.Append(true)
		for dimension := 0; dimension < nativeSegmentDimension; dimension++ {
			valuesBuilder.Append(float32(id) + float32(dimension)*0.01)
		}
	}

	idArray := idBuilder.NewArray()
	vectorArray := vectorBuilder.NewArray()
	record := array.NewRecord(
		schema,
		[]arrow.Array{idArray, vectorArray},
		int64(count),
	)
	idArray.Release()
	vectorArray.Release()
	defer record.Release()
	require.NoError(t, table.Add(ctx, record, nil))
}

func prepareNativeSegmentModel(
	t *testing.T,
	ctx context.Context,
	native contracts.ITableNativeSegments,
	version uint64,
	fragmentIDs []uint32,
	config contracts.NativeIndexConfig,
	configDigest string,
	modelID string,
	artifactOutputURI *string,
) contracts.PreparedIndexModel {
	t.Helper()
	prepared, err := native.PrepareIndexModel(ctx, contracts.PrepareIndexModelRequest{
		WireVersion:         contracts.NativeSegmentWireVersion,
		DatasetVersion:      version,
		TrainingFragmentIDs: fragmentIDs,
		VectorColumn:        "vector",
		LogicalIndexName:    "native_vector_idx",
		IndexConfigDigest:   configDigest,
		IndexConfig:         config,
		ModelID:             modelID,
		ModelScope:          "macro-test",
		ArtifactOutputURI:   artifactOutputURI,
	})
	require.NoError(t, err)
	require.Equal(t, version, prepared.DatasetVersion)
	require.ElementsMatch(t, fragmentIDs, prepared.TrainingFragmentIDs)
	require.Equal(t, contracts.NativeRuntimeVersion, prepared.Model.Identity.RuntimeVersion)
	require.True(t, strings.HasPrefix(prepared.Model.Identity.ModelChecksum, "sha256:"))
	require.Len(t, strings.TrimPrefix(prepared.Model.Identity.ModelChecksum, "sha256:"), 64)
	return *prepared
}

func buildNativeSegment(
	t *testing.T,
	ctx context.Context,
	native contracts.ITableNativeSegments,
	version uint64,
	fragmentIDs []uint32,
	config contracts.NativeIndexConfig,
	model contracts.NativeIndexModel,
) contracts.NativeIndexSegment {
	t.Helper()
	segment, err := native.CreateIndexUncommitted(ctx, contracts.SegmentBuildRequest{
		WireVersion:       contracts.NativeSegmentWireVersion,
		DatasetVersion:    version,
		FragmentIDs:       fragmentIDs,
		VectorColumn:      "vector",
		LogicalIndexName:  "native_vector_idx",
		IndexConfigDigest: "sha256:native-vector-config-v1",
		IndexConfig:       config,
		Model:             model,
	})
	require.NoError(t, err)
	require.Equal(t, version, segment.DatasetVersion)
	require.ElementsMatch(t, fragmentIDs, segment.FragmentIDs)
	require.NotEmpty(t, segment.UUID)
	require.NotEmpty(t, segment.OpaqueMetadata)
	return *segment
}

func buildMergedNativeGeneration(
	t *testing.T,
	ctx context.Context,
	native contracts.ITableNativeSegments,
	version uint64,
	allFragmentIDs []uint32,
	leftIDs []uint32,
	rightIDs []uint32,
	config contracts.NativeIndexConfig,
	configDigest string,
	modelID string,
) (contracts.NativeIndexSegment, contracts.CommitExistingIndexSegmentsRequest) {
	t.Helper()
	prepared := prepareNativeSegmentModel(
		t,
		ctx,
		native,
		version,
		allFragmentIDs,
		config,
		configDigest,
		modelID,
		nil,
	)
	left := buildNativeSegment(t, ctx, native, version, leftIDs, config, prepared.Model)
	right := buildNativeSegment(t, ctx, native, version, rightIDs, config, prepared.Model)
	merged, err := native.MergeExistingIndexSegments(ctx, contracts.MergeExistingIndexSegmentsRequest{
		WireVersion:       contracts.NativeSegmentWireVersion,
		DatasetVersion:    version,
		FragmentIDs:       allFragmentIDs,
		VectorColumn:      "vector",
		LogicalIndexName:  "native_vector_idx",
		IndexConfigDigest: configDigest,
		IndexConfig:       config,
		ModelIdentity:     prepared.Model.Identity,
		Segments:          []contracts.NativeIndexSegment{left, right},
	})
	require.NoError(t, err)
	require.ElementsMatch(t, allFragmentIDs, merged.FragmentIDs)
	return *merged, contracts.CommitExistingIndexSegmentsRequest{
		WireVersion:       contracts.NativeSegmentWireVersion,
		DatasetVersion:    version,
		FragmentIDs:       append([]uint32(nil), allFragmentIDs...),
		VectorColumn:      "vector",
		LogicalIndexName:  "native_vector_idx",
		IndexConfigDigest: configDigest,
		IndexConfig:       config,
		Segments:          []contracts.NativeIndexSegment{*merged},
	}
}

func TestNativeIndexSegmentsDistributedLifecycle(t *testing.T) {
	ctx := context.Background()
	connection, err := lancedb.Connect(ctx, t.TempDir(), nil)
	require.NoError(t, err)
	defer connection.Close()

	arrowSchema := arrow.NewSchema([]arrow.Field{
		{Name: "id", Type: arrow.PrimitiveTypes.Int32, Nullable: false},
		{Name: "vector", Type: arrow.FixedSizeListOf(nativeSegmentDimension, arrow.PrimitiveTypes.Float32), Nullable: false},
	}, nil)
	schema, err := internal.NewSchema(arrowSchema)
	require.NoError(t, err)
	table, err := connection.CreateTable(ctx, "native_segments", schema)
	require.NoError(t, err)
	defer table.Close()

	// Each write transaction creates a separate fragment.
	for fragment := 0; fragment < 4; fragment++ {
		appendNativeSegmentRows(t, ctx, table, arrowSchema, int32(fragment*64), 64)
	}
	version, err := table.Version(ctx)
	require.NoError(t, err)
	require.Positive(t, version)

	native, ok := table.(contracts.ITableNativeSegments)
	require.True(t, ok, "native table must implement ITableNativeSegments")
	fragments, err := native.ListFragments(ctx, uint64(version))
	require.NoError(t, err)
	require.Len(t, fragments, 4)
	allFragmentIDs := make([]uint32, len(fragments))
	for index, fragment := range fragments {
		require.Equal(t, uint64(64), fragment.RowCount)
		allFragmentIDs[index] = uint32(fragment.ID)
	}
	leftIDs := append([]uint32(nil), allFragmentIDs[:2]...)
	rightIDs := append([]uint32(nil), allFragmentIDs[2:]...)

	config := contracts.NativeIndexConfig{
		Type:          "IVF_FLAT",
		DistanceType:  "l2",
		Dimension:     nativeSegmentDimension,
		NumPartitions: 2,
	}
	const configDigest = "sha256:native-vector-config-v1"
	artifactDirectory := t.TempDir()
	shared := prepareNativeSegmentModel(
		t,
		ctx,
		native,
		uint64(version),
		allFragmentIDs,
		config,
		configDigest,
		"native-segment-model-shared",
		&artifactDirectory,
	)
	require.Empty(t, shared.Model.Centroids.Data)
	require.NotNil(t, shared.Model.Centroids.Reference)
	require.Equal(t, []uint32{2, nativeSegmentDimension}, shared.Model.Centroids.Reference.Shape)
	_, err = os.Stat(shared.Model.Centroids.Reference.URI)
	require.NoError(t, err, "Rust must atomically publish a readable local artifact")

	// Inline artifacts remain available for small models and receive a distinct
	// Rust-computed identity even if they train on the same rows.
	alternate := prepareNativeSegmentModel(
		t,
		ctx,
		native,
		uint64(version),
		allFragmentIDs,
		config,
		configDigest,
		"native-segment-model-alternate",
		nil,
	)
	require.NotEmpty(t, alternate.Model.Centroids.Data)
	require.Nil(t, alternate.Model.Centroids.Reference)
	require.NotEqual(t, shared.Model.Identity.ModelChecksum, alternate.Model.Identity.ModelChecksum)

	sharedLeft := buildNativeSegment(t, ctx, native, uint64(version), leftIDs, config, shared.Model)
	sharedRight := buildNativeSegment(t, ctx, native, uint64(version), rightIDs, config, shared.Model)
	alternateRight := buildNativeSegment(t, ctx, native, uint64(version), rightIDs, config, alternate.Model)

	mergeRequest := contracts.MergeExistingIndexSegmentsRequest{
		WireVersion:       contracts.NativeSegmentWireVersion,
		DatasetVersion:    uint64(version),
		FragmentIDs:       allFragmentIDs,
		VectorColumn:      "vector",
		LogicalIndexName:  "native_vector_idx",
		IndexConfigDigest: configDigest,
		IndexConfig:       config,
		ModelIdentity:     shared.Model.Identity,
		Segments:          []contracts.NativeIndexSegment{sharedLeft, alternateRight},
	}
	_, err = native.MergeExistingIndexSegments(ctx, mergeRequest)
	require.ErrorContains(t, err, "model identity is incompatible")

	// Two Micro Groups use the one prepared model and physically merge to one Macro Segment.
	mergeRequest.Segments = []contracts.NativeIndexSegment{sharedLeft, sharedRight}
	merged, err := native.MergeExistingIndexSegments(ctx, mergeRequest)
	require.NoError(t, err)
	require.ElementsMatch(t, allFragmentIDs, merged.FragmentIDs)

	commitRequest := contracts.CommitExistingIndexSegmentsRequest{
		WireVersion:       contracts.NativeSegmentWireVersion,
		DatasetVersion:    uint64(version),
		FragmentIDs:       allFragmentIDs,
		VectorColumn:      "vector",
		LogicalIndexName:  "native_vector_idx",
		IndexConfigDigest: configDigest,
		IndexConfig:       config,
		Segments:          []contracts.NativeIndexSegment{*merged},
	}
	overlap := commitRequest
	overlap.Segments = []contracts.NativeIndexSegment{sharedLeft, sharedLeft, sharedRight}
	_, err = native.CommitExistingIndexSegments(ctx, overlap)
	require.ErrorContains(t, err, "coverage overlaps")

	omission := commitRequest
	omission.Segments = []contracts.NativeIndexSegment{sharedLeft}
	_, err = native.CommitExistingIndexSegments(ctx, omission)
	require.ErrorContains(t, err, "coverage mismatch")

	invalidBuild := contracts.SegmentBuildRequest{
		WireVersion:       contracts.NativeSegmentWireVersion,
		DatasetVersion:    uint64(version),
		FragmentIDs:       []uint32{allFragmentIDs[len(allFragmentIDs)-1] + 1000},
		VectorColumn:      "vector",
		LogicalIndexName:  "native_vector_idx",
		IndexConfigDigest: configDigest,
		IndexConfig:       config,
		Model:             shared.Model,
	}
	_, err = native.CreateIndexUncommitted(ctx, invalidBuild)
	require.ErrorContains(t, err, "not present in dataset version")

	tamperedModel := shared.Model
	tamperedModel.Identity.ModelChecksum = "sha256:" + strings.Repeat("0", 64)
	tamperedBuild := invalidBuild
	tamperedBuild.FragmentIDs = leftIDs
	tamperedBuild.Model = tamperedModel
	_, err = native.CreateIndexUncommitted(ctx, tamperedBuild)
	require.ErrorContains(t, err, "model checksum mismatch")

	invalidPrepare := contracts.PrepareIndexModelRequest{
		WireVersion:         contracts.NativeSegmentWireVersion,
		DatasetVersion:      uint64(version),
		TrainingFragmentIDs: []uint32{allFragmentIDs[len(allFragmentIDs)-1] + 1000},
		VectorColumn:        "vector",
		LogicalIndexName:    "native_vector_idx",
		IndexConfigDigest:   configDigest,
		IndexConfig:         config,
		ModelID:             "invalid-fragment-model",
		ModelScope:          "macro-test",
	}
	_, err = native.PrepareIndexModel(ctx, invalidPrepare)
	require.ErrorContains(t, err, "not present in dataset version")
	invalidPrepare.DatasetVersion = uint64(version) + 1000
	invalidPrepare.TrainingFragmentIDs = allFragmentIDs
	_, err = native.PrepareIndexModel(ctx, invalidPrepare)
	require.Error(t, err, "a non-existent fixed dataset version must fail")

	// Exercise Lance 8's scoped PQ trainer as well as the IVF trainer.
	bits, subVectors, maxIterations, sampleRate := uint32(4), uint32(2), uint32(3), uint32(16)
	pqConfig := contracts.NativeIndexConfig{
		Type:          "IVF_PQ",
		DistanceType:  "l2",
		Dimension:     nativeSegmentDimension,
		NumPartitions: 2,
		NumSubVectors: &subVectors,
		NumBits:       &bits,
		MaxIterations: &maxIterations,
		SampleRate:    &sampleRate,
	}
	pqPrepared := prepareNativeSegmentModel(
		t,
		ctx,
		native,
		uint64(version),
		allFragmentIDs,
		pqConfig,
		"sha256:native-pq-config-v1",
		"native-segment-pq-model",
		nil,
	)
	require.NotNil(t, pqPrepared.Model.PQCodebook)
	require.NotEmpty(t, pqPrepared.Model.PQCodebook.Data)
	require.Equal(t, []uint32{2, 16, 2}, pqPrepared.Model.PQCodebook.Shape)

	committed, err := native.CommitExistingIndexSegments(ctx, commitRequest)
	require.NoError(t, err)
	require.Greater(t, committed.CommittedDatasetVersion, uint64(version))
	require.Len(t, committed.SegmentUUIDs, 1)

	firstInspection, err := native.InspectIndexSegments(ctx, "native_vector_idx")
	require.NoError(t, err)
	require.Equal(t, "native_vector_idx", firstInspection.LogicalIndexName)
	require.NotNil(t, firstInspection.VectorColumn)
	require.Equal(t, "vector", *firstInspection.VectorColumn)
	require.Len(t, firstInspection.Segments, 1)
	require.Equal(t, merged.UUID, firstInspection.Segments[0].UUID)
	firstUUID := firstInspection.Segments[0].UUID

	results, err := table.VectorSearch(ctx, "vector", []float32{0, 0.01, 0.02, 0.03}, 3)
	require.NoError(t, err)
	require.Len(t, results, 3)
	require.Equal(t, fmt.Sprint(0), fmt.Sprint(results[0]["id"]))

	// Plan a complete second generation from the first commit's latest version.
	// The same logical name must be accepted while the first generation remains
	// queryable until the replacement commit.
	secondSourceVersion, err := table.Version(ctx)
	require.NoError(t, err)
	require.Equal(t, committed.CommittedDatasetVersion, uint64(secondSourceVersion))
	secondFragments, err := native.ListFragments(ctx, uint64(secondSourceVersion))
	require.NoError(t, err)
	require.Equal(t, fragments, secondFragments)
	secondMerged, secondCommit := buildMergedNativeGeneration(
		t,
		ctx,
		native,
		uint64(secondSourceVersion),
		allFragmentIDs,
		leftIDs,
		rightIDs,
		config,
		configDigest,
		"native-segment-model-rebuild",
	)

	beforeSecondCommit, err := native.InspectIndexSegments(ctx, "native_vector_idx")
	require.NoError(t, err)
	require.Len(t, beforeSecondCommit.Segments, 1)
	require.Equal(t, firstUUID, beforeSecondCommit.Segments[0].UUID)
	results, err = table.VectorSearch(ctx, "vector", []float32{0, 0.01, 0.02, 0.03}, 3)
	require.NoError(t, err)
	require.Equal(t, fmt.Sprint(0), fmt.Sprint(results[0]["id"]))

	secondCommitted, err := native.CommitExistingIndexSegments(ctx, secondCommit)
	require.NoError(t, err)
	require.Greater(t, secondCommitted.CommittedDatasetVersion, uint64(secondSourceVersion))
	require.Equal(t, []string{secondMerged.UUID}, secondCommitted.SegmentUUIDs)
	secondInspection, err := native.InspectIndexSegments(ctx, "native_vector_idx")
	require.NoError(t, err)
	require.Len(t, secondInspection.Segments, 1)
	require.Equal(t, secondMerged.UUID, secondInspection.Segments[0].UUID)
	require.NotEqual(t, firstUUID, secondInspection.Segments[0].UUID)
	results, err = table.VectorSearch(ctx, "vector", []float32{0, 0.01, 0.02, 0.03}, 3)
	require.NoError(t, err)
	require.Len(t, results, 3)
	require.Equal(t, fmt.Sprint(0), fmt.Sprint(results[0]["id"]))

	// Stage another same-name generation, then drift the manifest with an append.
	// The stale commit must fail without changing the second generation.
	driftSourceVersion, err := table.Version(ctx)
	require.NoError(t, err)
	_, driftCommit := buildMergedNativeGeneration(
		t,
		ctx,
		native,
		uint64(driftSourceVersion),
		allFragmentIDs,
		leftIDs,
		rightIDs,
		config,
		configDigest,
		"native-segment-model-stale",
	)

	appendNativeSegmentRows(t, ctx, table, arrowSchema, 10_000, 64)
	latestVersion, err := table.Version(ctx)
	require.NoError(t, err)
	require.Greater(t, latestVersion, driftSourceVersion)
	_, err = native.CommitExistingIndexSegments(ctx, driftCommit)
	require.ErrorContains(t, err, "dataset version mismatch")
	afterRejectedCommit, err := native.InspectIndexSegments(ctx, "native_vector_idx")
	require.NoError(t, err)
	require.Len(t, afterRejectedCommit.Segments, 1)
	require.Equal(t, secondMerged.UUID, afterRejectedCommit.Segments[0].UUID)
	results, err = table.VectorSearch(ctx, "vector", []float32{0, 0.01, 0.02, 0.03}, 3)
	require.NoError(t, err)
	require.Equal(t, fmt.Sprint(0), fmt.Sprint(results[0]["id"]))

	// Historical enumeration and training remain pinned even with a newer fragment.
	latestFragments, err := native.ListFragments(ctx, uint64(latestVersion))
	require.NoError(t, err)
	require.Len(t, latestFragments, 5)
	pinned := prepareNativeSegmentModel(
		t,
		ctx,
		native,
		uint64(version),
		allFragmentIDs,
		config,
		configDigest,
		"historical-snapshot-model",
		nil,
	)
	require.ElementsMatch(t, allFragmentIDs, pinned.TrainingFragmentIDs)

	historical, err := native.ListFragments(ctx, uint64(version))
	require.NoError(t, err)
	require.Equal(t, fragments, historical)
}
