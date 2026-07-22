// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

package internal

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#include "lancedb.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"unsafe"

	"github.com/eozsahin1993/lancedb-go/pkg/contracts"
)

var _ contracts.ITableNativeSegments = (*Table)(nil)

func nativeSegmentResultError(operation string, result *C.SimpleResult) error {
	if result == nil {
		return fmt.Errorf("%s failed: native FFI returned a nil result", operation)
	}
	message := "unknown error"
	if result.ERROR_MESSAGE != nil {
		message = C.GoString(result.ERROR_MESSAGE)
	}
	runtimeVersion := "unknown"
	if result.RUNTIME_VERSION != nil {
		runtimeVersion = C.GoString(result.RUNTIME_VERSION)
	}
	return fmt.Errorf(
		"%s failed (ffi_code=%d, runtime=%s): %s",
		operation,
		int(result.ERROR_CODE),
		runtimeVersion,
		message,
	)
}

func decodeNativeSegmentResponse(output *C.uchar, outputLen C.size_t, target any) error {
	if output == nil {
		return fmt.Errorf("native FFI returned an empty response")
	}
	defer C.simple_lancedb_free_bytes(output, outputLen)
	if uint64(outputLen) > math.MaxInt32 {
		return fmt.Errorf("native FFI response exceeds Go C.GoBytes limit: %d bytes", uint64(outputLen))
	}
	data := C.GoBytes(unsafe.Pointer(output), C.int(outputLen))
	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("failed to decode native segment response: %w", err)
	}
	return nil
}

func marshalNativeSegmentRequest(request any) ([]byte, error) {
	data, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to encode native segment request: %w", err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("native segment request encoded to an empty payload")
	}
	return data, nil
}

func validateNativeSegmentWire(operation string, wireVersion uint32, runtimeVersion string) error {
	if wireVersion != contracts.NativeSegmentWireVersion {
		return fmt.Errorf(
			"%s response wire version mismatch: got %d, expected %d",
			operation,
			wireVersion,
			contracts.NativeSegmentWireVersion,
		)
	}
	if runtimeVersion != contracts.NativeRuntimeVersion {
		return fmt.Errorf(
			"%s response runtime mismatch: got %q, expected %q",
			operation,
			runtimeVersion,
			contracts.NativeRuntimeVersion,
		)
	}
	return nil
}

func (t *Table) nativeSegmentReadLock(ctx context.Context) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	t.mu.RLock()
	if t.closed || t.handle == nil {
		t.mu.RUnlock()
		return nil, fmt.Errorf("table is closed")
	}
	return t.mu.RUnlock, nil
}

func (t *Table) nativeSegmentWriteLock(ctx context.Context) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	t.mu.Lock()
	if t.closed || t.handle == nil {
		t.mu.Unlock()
		return nil, fmt.Errorf("table is closed")
	}
	return t.mu.Unlock, nil
}

// ListFragments enumerates fragments from exactly datasetVersion.
func (t *Table) ListFragments(ctx context.Context, datasetVersion uint64) ([]contracts.FragmentInfo, error) {
	unlock, err := t.nativeSegmentReadLock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()

	var output *C.uchar
	var outputLen C.size_t
	result := C.simple_lancedb_table_list_fragments(
		t.handle,
		C.uint64_t(datasetVersion),
		&output,
		&outputLen,
	)
	defer C.simple_lancedb_result_free(result)
	if result == nil || !result.SUCCESS {
		return nil, nativeSegmentResultError("list_fragments", result)
	}
	var response struct {
		WireVersion    uint32                   `json:"wire_version"`
		RuntimeVersion string                   `json:"runtime_version"`
		DatasetVersion uint64                   `json:"dataset_version"`
		Fragments      []contracts.FragmentInfo `json:"fragments"`
	}
	if err := decodeNativeSegmentResponse(output, outputLen, &response); err != nil {
		return nil, err
	}
	if err := validateNativeSegmentWire("list_fragments", response.WireVersion, response.RuntimeVersion); err != nil {
		return nil, err
	}
	if response.DatasetVersion != datasetVersion {
		return nil, fmt.Errorf(
			"list_fragments response dataset version mismatch: got %d, expected %d",
			response.DatasetVersion,
			datasetVersion,
		)
	}
	return response.Fragments, nil
}

// PrepareIndexModel trains a shared IVF / IVF_PQ model on an exact snapshot.
func (t *Table) PrepareIndexModel(ctx context.Context, request contracts.PrepareIndexModelRequest) (*contracts.PreparedIndexModel, error) {
	payload, err := marshalNativeSegmentRequest(request)
	if err != nil {
		return nil, err
	}
	unlock, err := t.nativeSegmentReadLock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()

	requestBytes := C.CBytes(payload)
	defer C.free(requestBytes)
	var output *C.uchar
	var outputLen C.size_t
	result := C.simple_lancedb_table_prepare_index_model(
		t.handle,
		(*C.uchar)(requestBytes),
		C.size_t(len(payload)),
		&output,
		&outputLen,
	)
	defer C.simple_lancedb_result_free(result)
	if result == nil || !result.SUCCESS {
		return nil, nativeSegmentResultError("prepare_index_model", result)
	}
	var response contracts.PreparedIndexModel
	if err := decodeNativeSegmentResponse(output, outputLen, &response); err != nil {
		return nil, err
	}
	if err := validateNativeSegmentWire("prepare_index_model", response.WireVersion, response.RuntimeVersion); err != nil {
		return nil, err
	}
	if response.DatasetVersion != request.DatasetVersion {
		return nil, fmt.Errorf(
			"prepare_index_model response dataset version mismatch: got %d, expected %d",
			response.DatasetVersion,
			request.DatasetVersion,
		)
	}
	if !slices.Equal(response.TrainingFragmentIDs, request.TrainingFragmentIDs) ||
		response.VectorColumn != request.VectorColumn ||
		response.LogicalIndexName != request.LogicalIndexName ||
		response.IndexConfigDigest != request.IndexConfigDigest ||
		response.Model.Identity.ModelID != request.ModelID ||
		response.Model.Identity.ModelScope != request.ModelScope {
		return nil, fmt.Errorf("prepare_index_model response context does not match the request")
	}
	return &response, nil
}

// CreateIndexUncommitted builds an uncommitted physical vector segment.
func (t *Table) CreateIndexUncommitted(ctx context.Context, request contracts.SegmentBuildRequest) (*contracts.NativeIndexSegment, error) {
	payload, err := marshalNativeSegmentRequest(request)
	if err != nil {
		return nil, err
	}
	unlock, err := t.nativeSegmentReadLock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()

	requestBytes := C.CBytes(payload)
	defer C.free(requestBytes)
	var output *C.uchar
	var outputLen C.size_t
	result := C.simple_lancedb_table_create_index_uncommitted(
		t.handle,
		(*C.uchar)(requestBytes),
		C.size_t(len(payload)),
		&output,
		&outputLen,
	)
	defer C.simple_lancedb_result_free(result)
	if result == nil || !result.SUCCESS {
		return nil, nativeSegmentResultError("create_index_uncommitted", result)
	}
	var response contracts.NativeIndexSegment
	if err := decodeNativeSegmentResponse(output, outputLen, &response); err != nil {
		return nil, err
	}
	if err := validateNativeSegmentWire("create_index_uncommitted", response.WireVersion, response.RuntimeVersion); err != nil {
		return nil, err
	}
	return &response, nil
}

// MergeExistingIndexSegments physically merges model-compatible segments.
func (t *Table) MergeExistingIndexSegments(ctx context.Context, request contracts.MergeExistingIndexSegmentsRequest) (*contracts.NativeIndexSegment, error) {
	payload, err := marshalNativeSegmentRequest(request)
	if err != nil {
		return nil, err
	}
	unlock, err := t.nativeSegmentReadLock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()

	requestBytes := C.CBytes(payload)
	defer C.free(requestBytes)
	var output *C.uchar
	var outputLen C.size_t
	result := C.simple_lancedb_table_merge_existing_index_segments(
		t.handle,
		(*C.uchar)(requestBytes),
		C.size_t(len(payload)),
		&output,
		&outputLen,
	)
	defer C.simple_lancedb_result_free(result)
	if result == nil || !result.SUCCESS {
		return nil, nativeSegmentResultError("merge_existing_index_segments", result)
	}
	var response contracts.NativeIndexSegment
	if err := decodeNativeSegmentResponse(output, outputLen, &response); err != nil {
		return nil, err
	}
	if err := validateNativeSegmentWire("merge_existing_index_segments", response.WireVersion, response.RuntimeVersion); err != nil {
		return nil, err
	}
	return &response, nil
}

// CommitExistingIndexSegments publishes physical segments as one logical index.
func (t *Table) CommitExistingIndexSegments(ctx context.Context, request contracts.CommitExistingIndexSegmentsRequest) (*contracts.CommittedIndex, error) {
	payload, err := marshalNativeSegmentRequest(request)
	if err != nil {
		return nil, err
	}
	unlock, err := t.nativeSegmentWriteLock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()

	requestBytes := C.CBytes(payload)
	defer C.free(requestBytes)
	var output *C.uchar
	var outputLen C.size_t
	result := C.simple_lancedb_table_commit_existing_index_segments(
		t.handle,
		(*C.uchar)(requestBytes),
		C.size_t(len(payload)),
		&output,
		&outputLen,
	)
	defer C.simple_lancedb_result_free(result)
	if result == nil || !result.SUCCESS {
		return nil, nativeSegmentResultError("commit_existing_index_segments", result)
	}
	var response contracts.CommittedIndex
	if err := decodeNativeSegmentResponse(output, outputLen, &response); err != nil {
		return nil, err
	}
	if err := validateNativeSegmentWire("commit_existing_index_segments", response.WireVersion, response.RuntimeVersion); err != nil {
		return nil, err
	}
	return &response, nil
}

// InspectIndexSegments returns committed physical segment metadata.
func (t *Table) InspectIndexSegments(ctx context.Context, indexName string) (*contracts.IndexSegmentsInspection, error) {
	unlock, err := t.nativeSegmentReadLock(ctx)
	if err != nil {
		return nil, err
	}
	defer unlock()

	cIndexName := C.CString(indexName)
	defer C.free(unsafe.Pointer(cIndexName))
	var output *C.uchar
	var outputLen C.size_t
	result := C.simple_lancedb_table_inspect_index_segments(
		t.handle,
		cIndexName,
		&output,
		&outputLen,
	)
	defer C.simple_lancedb_result_free(result)
	if result == nil || !result.SUCCESS {
		return nil, nativeSegmentResultError("inspect_index_segments", result)
	}
	var response contracts.IndexSegmentsInspection
	if err := decodeNativeSegmentResponse(output, outputLen, &response); err != nil {
		return nil, err
	}
	if err := validateNativeSegmentWire("inspect_index_segments", response.WireVersion, response.RuntimeVersion); err != nil {
		return nil, err
	}
	if response.LogicalIndexName != indexName {
		return nil, fmt.Errorf(
			"inspect_index_segments response index mismatch: got %q, expected %q",
			response.LogicalIndexName,
			indexName,
		)
	}
	for index := range response.Segments {
		if err := validateNativeSegmentWire(
			fmt.Sprintf("inspect_index_segments.segments[%d]", index),
			response.Segments[index].WireVersion,
			response.Segments[index].RuntimeVersion,
		); err != nil {
			return nil, err
		}
	}
	return &response, nil
}
