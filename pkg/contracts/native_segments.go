// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

package contracts

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
)

const (
	// NativeSegmentWireVersion versions the SDK-owned JSON/bytes contract.
	NativeSegmentWireVersion uint32 = 1
	// NativeRuntimeVersion is embedded in every request model identity and response.
	NativeRuntimeVersion = "lancedb=0.31.0;lance=8.0.0;arrow=58.0.0;rust=1.91.0;native-segment-wire=1"
)

// ITableNativeSegments is an optional native-table capability. It stays out of
// ITable so existing mocks and downstream implementations remain source compatible.
type ITableNativeSegments interface {
	ListFragments(ctx context.Context, datasetVersion uint64) ([]FragmentInfo, error)
	PrepareIndexModel(ctx context.Context, request PrepareIndexModelRequest) (*PreparedIndexModel, error)
	CreateIndexUncommitted(ctx context.Context, request SegmentBuildRequest) (*NativeIndexSegment, error)
	MergeExistingIndexSegments(ctx context.Context, request MergeExistingIndexSegmentsRequest) (*NativeIndexSegment, error)
	CommitExistingIndexSegments(ctx context.Context, request CommitExistingIndexSegmentsRequest) (*CommittedIndex, error)
	InspectIndexSegments(ctx context.Context, indexName string) (*IndexSegmentsInspection, error)
}

// FragmentInfo describes one fragment in an exact dataset snapshot.
type FragmentInfo struct {
	ID               uint64  `json:"id"`
	RowCount         uint64  `json:"row_count"`
	PhysicalRowCount uint64  `json:"physical_row_count"`
	DataBytes        uint64  `json:"data_bytes"`
	DeletionRate     float64 `json:"deletion_rate"`
}

// NativeIndexConfig is the stable distributed-vector build configuration.
// Native segment builds currently support IVF_FLAT and IVF_PQ.
type NativeIndexConfig struct {
	Type                string  `json:"type"`
	DistanceType        string  `json:"distance_type"`
	Dimension           uint32  `json:"dimension"`
	NumPartitions       uint32  `json:"num_partitions"`
	NumSubVectors       *uint32 `json:"num_sub_vectors,omitempty"`
	NumBits             *uint32 `json:"num_bits,omitempty"`
	MaxIterations       *uint32 `json:"max_iterations,omitempty"`
	SampleRate          *uint32 `json:"sample_rate,omitempty"`
	TargetPartitionSize *uint32 `json:"target_partition_size,omitempty"`
}

// ModelIdentity determines physical-merge compatibility. Segments with
// different identities can be committed together, but cannot be physically merged.
type ModelIdentity struct {
	ModelID        string `json:"model_id"`
	ModelChecksum  string `json:"model_checksum"`
	ModelScope     string `json:"model_scope"`
	RuntimeVersion string `json:"runtime_version"`
}

// ModelArtifactReference lets Rust load a model tensor directly from local or
// shared object storage. Checksum covers the raw little-endian tensor bytes.
type ModelArtifactReference struct {
	URI            string   `json:"uri"`
	DataType       string   `json:"data_type"`
	Shape          []uint32 `json:"shape"`
	Checksum       string   `json:"checksum"`
	ByteLength     uint64   `json:"byte_length"`
	RuntimeVersion string   `json:"runtime_version"`
}

// Float32Artifact carries exactly one of inline Data or Reference. encoding/json
// serializes inline Data as base64; production-sized models should use Reference.
type Float32Artifact struct {
	DataType  string                  `json:"data_type"`
	Shape     []uint32                `json:"shape"`
	Data      []byte                  `json:"data,omitempty"`
	Reference *ModelArtifactReference `json:"reference,omitempty"`
}

// NewFloat32Artifact creates a validated float32_le model tensor.
func NewFloat32Artifact(shape []uint32, values []float32) (Float32Artifact, error) {
	if len(shape) == 0 {
		return Float32Artifact{}, fmt.Errorf("shape must not be empty")
	}
	elements := uint64(1)
	for _, dimension := range shape {
		if dimension == 0 {
			return Float32Artifact{}, fmt.Errorf("shape dimensions must be positive")
		}
		if elements > math.MaxUint64/uint64(dimension) {
			return Float32Artifact{}, fmt.Errorf("shape element count overflows uint64")
		}
		elements *= uint64(dimension)
	}
	if elements != uint64(len(values)) {
		return Float32Artifact{}, fmt.Errorf("shape requires %d values, got %d", elements, len(values))
	}
	data := make([]byte, len(values)*4)
	for index, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return Float32Artifact{}, fmt.Errorf("values[%d] is not finite", index)
		}
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return Float32Artifact{
		DataType: "float32_le",
		Shape:    append([]uint32(nil), shape...),
		Data:     data,
	}, nil
}

// NativeIndexModel supplies the precomputed IVF model required by Lance's
// distributed build. IVF_PQ additionally requires PQCodebook.
type NativeIndexModel struct {
	Identity   ModelIdentity    `json:"identity"`
	Centroids  Float32Artifact  `json:"centroids"`
	PQCodebook *Float32Artifact `json:"pq_codebook,omitempty"`
}

// PrepareIndexModelRequest trains a model against exactly one dataset snapshot
// and explicit training fragment set. ModelChecksum is intentionally absent:
// Rust computes it from the trained artifacts and canonical configuration.
type PrepareIndexModelRequest struct {
	WireVersion            uint32            `json:"wire_version"`
	DatasetVersion         uint64            `json:"dataset_version"`
	TrainingFragmentIDs    []uint32          `json:"training_fragment_ids"`
	VectorColumn           string            `json:"vector_column"`
	LogicalIndexName       string            `json:"logical_index_name"`
	IndexConfigDigest      string            `json:"index_config_digest"`
	IndexConfig            NativeIndexConfig `json:"index_config"`
	ModelID                string            `json:"model_id"`
	ModelScope             string            `json:"model_scope"`
	ArtifactOutputURI      *string           `json:"artifact_output_uri,omitempty"`
	InlineArtifactMaxBytes *uint64           `json:"inline_artifact_max_bytes,omitempty"`
}

// PreparedIndexModel contains Rust-trained artifacts and their canonical identity.
type PreparedIndexModel struct {
	WireVersion         uint32            `json:"wire_version"`
	RuntimeVersion      string            `json:"runtime_version"`
	DatasetVersion      uint64            `json:"dataset_version"`
	TrainingFragmentIDs []uint32          `json:"training_fragment_ids"`
	VectorColumn        string            `json:"vector_column"`
	LogicalIndexName    string            `json:"logical_index_name"`
	IndexConfigDigest   string            `json:"index_config_digest"`
	IndexConfig         NativeIndexConfig `json:"index_config"`
	Model               NativeIndexModel  `json:"model"`
}

// NativeIndexDetails is the stable Any envelope. Value remains opaque to Go.
type NativeIndexDetails struct {
	TypeURL string `json:"type_url"`
	Value   []byte `json:"value"`
}

// NativeIndexSegment is the SDK-owned versioned representation of a physical
// segment. OpaqueMetadata contains the complete Lance IndexMetadata protobuf
// needed by later merge and commit calls.
type NativeIndexSegment struct {
	WireVersion       uint32             `json:"wire_version"`
	RuntimeVersion    string             `json:"runtime_version"`
	UUID              string             `json:"uuid"`
	LogicalIndexName  string             `json:"logical_index_name"`
	DatasetVersion    uint64             `json:"dataset_version"`
	FragmentIDs       []uint32           `json:"fragment_ids"`
	VectorColumn      string             `json:"vector_column"`
	IndexVersion      int32              `json:"index_version"`
	IndexDetails      NativeIndexDetails `json:"index_details"`
	OpaqueMetadata    []byte             `json:"opaque_metadata"`
	IndexConfigDigest *string            `json:"index_config_digest,omitempty"`
	IndexConfig       *NativeIndexConfig `json:"index_config,omitempty"`
	ModelIdentity     *ModelIdentity     `json:"model_identity,omitempty"`
}

// SegmentBuildRequest always names one fixed snapshot, explicit fragment
// coverage, vector column, complete index config, and model identity/artifacts.
type SegmentBuildRequest struct {
	WireVersion       uint32            `json:"wire_version"`
	DatasetVersion    uint64            `json:"dataset_version"`
	FragmentIDs       []uint32          `json:"fragment_ids"`
	VectorColumn      string            `json:"vector_column"`
	LogicalIndexName  string            `json:"logical_index_name"`
	IndexConfigDigest string            `json:"index_config_digest"`
	IndexConfig       NativeIndexConfig `json:"index_config"`
	Model             NativeIndexModel  `json:"model"`
}

// MergeExistingIndexSegmentsRequest physically merges only segments that have
// exactly the same model identity and index configuration.
type MergeExistingIndexSegmentsRequest struct {
	WireVersion       uint32               `json:"wire_version"`
	DatasetVersion    uint64               `json:"dataset_version"`
	FragmentIDs       []uint32             `json:"fragment_ids"`
	VectorColumn      string               `json:"vector_column"`
	LogicalIndexName  string               `json:"logical_index_name"`
	IndexConfigDigest string               `json:"index_config_digest"`
	IndexConfig       NativeIndexConfig    `json:"index_config"`
	ModelIdentity     ModelIdentity        `json:"model_identity"`
	Segments          []NativeIndexSegment `json:"segments"`
}

// CommitExistingIndexSegmentsRequest atomically publishes all physical
// segments as one logical vector index. Different model identities are allowed;
// overlap, omission, config drift, and dataset-version drift are rejected.
type CommitExistingIndexSegmentsRequest struct {
	WireVersion       uint32               `json:"wire_version"`
	DatasetVersion    uint64               `json:"dataset_version"`
	FragmentIDs       []uint32             `json:"fragment_ids"`
	VectorColumn      string               `json:"vector_column"`
	LogicalIndexName  string               `json:"logical_index_name"`
	IndexConfigDigest string               `json:"index_config_digest"`
	IndexConfig       NativeIndexConfig    `json:"index_config"`
	Segments          []NativeIndexSegment `json:"segments"`
}

// CommittedIndex reports the manifest version created by a segment commit.
type CommittedIndex struct {
	WireVersion             uint32   `json:"wire_version"`
	RuntimeVersion          string   `json:"runtime_version"`
	LogicalIndexName        string   `json:"logical_index_name"`
	SourceDatasetVersion    uint64   `json:"source_dataset_version"`
	CommittedDatasetVersion uint64   `json:"committed_dataset_version"`
	VectorColumn            string   `json:"vector_column"`
	FragmentIDs             []uint32 `json:"fragment_ids"`
	SegmentUUIDs            []string `json:"segment_uuids"`
}

// IndexSegmentsInspection describes committed physical segments without
// exposing Lance's unstable Rust metadata layout.
type IndexSegmentsInspection struct {
	WireVersion      uint32               `json:"wire_version"`
	RuntimeVersion   string               `json:"runtime_version"`
	DatasetVersion   uint64               `json:"dataset_version"`
	LogicalIndexName string               `json:"logical_index_name"`
	VectorColumn     *string              `json:"vector_column,omitempty"`
	Segments         []NativeIndexSegment `json:"segments"`
}
