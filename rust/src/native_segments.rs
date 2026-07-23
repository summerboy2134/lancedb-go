// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Versioned Native Index Segment wire contract and C ABI.
//!
//! The C boundary exchanges UTF-8 JSON as length-delimited bytes.  Stable fields
//! are represented explicitly while the complete Lance `IndexMetadata` protobuf
//! is carried as opaque base64.  The latter is decoded only by this Rust layer.

use std::collections::{BTreeSet, HashSet};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array};
use arrow_schema::DataType;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use lance::index::vector::ivf::build_ivf_model;
use lance::index::vector::pq::build_pq_model_in_fragments;
use lance::index::vector::VectorIndexParams;
use lance::index::DatasetIndexExt;
use lance::Dataset;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::IndexType as LanceIndexType;
use lance_linalg::distance::DistanceType;
use lance_table::format::{pb, IndexMetadata};
use object_store::path::Path as ObjectStorePath;
use object_store::{parse_url_opts, ObjectStoreExt};
use prost::Message;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use url::Url;

use crate::ffi::{from_c_str, SimpleResult, NATIVE_RUNTIME_VERSION, NATIVE_SEGMENT_WIRE_VERSION};
use crate::runtime::get_simple_runtime;

const DEFAULT_INLINE_ARTIFACT_MAX_BYTES: usize = 1024 * 1024;
static ARTIFACT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeIndexConfig {
    #[serde(rename = "type")]
    pub index_type: String,
    pub distance_type: String,
    pub dimension: u32,
    pub num_partitions: u32,
    #[serde(default)]
    pub num_sub_vectors: Option<u32>,
    #[serde(default)]
    pub num_bits: Option<u32>,
    #[serde(default)]
    pub max_iterations: Option<u32>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
    #[serde(default)]
    pub target_partition_size: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_id: String,
    pub model_checksum: String,
    pub model_scope: String,
    pub runtime_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifactReference {
    pub uri: String,
    pub data_type: String,
    pub shape: Vec<u32>,
    pub checksum: String,
    pub byte_length: u64,
    pub runtime_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Float32Artifact {
    pub data_type: String,
    pub shape: Vec<u32>,
    /// Standard base64 of little-endian IEEE-754 f32 values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<ModelArtifactReference>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexModel {
    pub identity: ModelIdentity,
    pub centroids: Float32Artifact,
    #[serde(default)]
    pub pq_codebook: Option<Float32Artifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareIndexModelRequest {
    pub wire_version: u32,
    pub dataset_version: u64,
    pub training_fragment_ids: Vec<u32>,
    pub vector_column: String,
    pub logical_index_name: String,
    pub index_config_digest: String,
    pub index_config: NativeIndexConfig,
    pub model_id: String,
    pub model_scope: String,
    /// Directory/prefix where immutable model tensors are atomically published.
    #[serde(default)]
    pub artifact_output_uri: Option<String>,
    /// Inline response limit when artifact_output_uri is omitted.
    #[serde(default)]
    pub inline_artifact_max_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PreparedIndexModel {
    pub wire_version: u32,
    pub runtime_version: String,
    pub dataset_version: u64,
    pub training_fragment_ids: Vec<u32>,
    pub vector_column: String,
    pub logical_index_name: String,
    pub index_config_digest: String,
    pub index_config: NativeIndexConfig,
    pub model: IndexModel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDetailsEnvelope {
    pub type_url: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexSegmentEnvelope {
    pub wire_version: u32,
    pub runtime_version: String,
    pub uuid: String,
    pub logical_index_name: String,
    pub dataset_version: u64,
    pub fragment_ids: Vec<u32>,
    pub vector_column: String,
    pub index_version: i32,
    pub index_details: IndexDetailsEnvelope,
    pub opaque_metadata: String,
    #[serde(default)]
    pub index_config_digest: Option<String>,
    #[serde(default)]
    pub index_config: Option<NativeIndexConfig>,
    #[serde(default)]
    pub model_identity: Option<ModelIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentBuildRequest {
    pub wire_version: u32,
    pub dataset_version: u64,
    pub fragment_ids: Vec<u32>,
    pub vector_column: String,
    pub logical_index_name: String,
    pub index_config_digest: String,
    pub index_config: NativeIndexConfig,
    pub model: IndexModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeSegmentsRequest {
    pub wire_version: u32,
    pub dataset_version: u64,
    pub fragment_ids: Vec<u32>,
    pub vector_column: String,
    pub logical_index_name: String,
    pub index_config_digest: String,
    pub index_config: NativeIndexConfig,
    pub model_identity: ModelIdentity,
    pub segments: Vec<IndexSegmentEnvelope>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitSegmentsRequest {
    pub wire_version: u32,
    pub dataset_version: u64,
    pub fragment_ids: Vec<u32>,
    pub vector_column: String,
    pub logical_index_name: String,
    pub index_config_digest: String,
    pub index_config: NativeIndexConfig,
    pub segments: Vec<IndexSegmentEnvelope>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FragmentInfo {
    pub id: u64,
    pub row_count: u64,
    pub physical_row_count: u64,
    pub data_bytes: u64,
    pub deletion_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FragmentListResponse {
    pub wire_version: u32,
    pub runtime_version: String,
    pub dataset_version: u64,
    pub fragments: Vec<FragmentInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommitSegmentsResponse {
    pub wire_version: u32,
    pub runtime_version: String,
    pub logical_index_name: String,
    pub source_dataset_version: u64,
    pub committed_dataset_version: u64,
    pub vector_column: String,
    pub fragment_ids: Vec<u32>,
    pub segment_uuids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectIndexSegmentsResponse {
    pub wire_version: u32,
    pub runtime_version: String,
    pub dataset_version: u64,
    pub logical_index_name: String,
    pub vector_column: Option<String>,
    pub segments: Vec<IndexSegmentEnvelope>,
}

fn validate_wire_version(version: u32) -> Result<(), String> {
    if version != NATIVE_SEGMENT_WIRE_VERSION {
        return Err(format!(
            "unsupported native segment wire_version {version}; expected {NATIVE_SEGMENT_WIRE_VERSION}"
        ));
    }
    Ok(())
}

fn validate_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{field} must not be empty"))
    } else {
        Ok(())
    }
}

fn validate_model_identity(identity: &ModelIdentity) -> Result<(), String> {
    validate_non_empty(&identity.model_id, "model_identity.model_id")?;
    validate_non_empty(&identity.model_checksum, "model_identity.model_checksum")?;
    validate_checksum_syntax(&identity.model_checksum, "model_identity.model_checksum")?;
    validate_non_empty(&identity.model_scope, "model_identity.model_scope")?;
    if identity.runtime_version != NATIVE_RUNTIME_VERSION {
        return Err(format!(
            "model runtime_version mismatch: got '{}', expected '{}'",
            identity.runtime_version, NATIVE_RUNTIME_VERSION
        ));
    }
    Ok(())
}

impl Float32Artifact {
    async fn load(&self, field: &str) -> Result<LoadedFloat32Artifact, String> {
        if self.data_type != "float32_le" {
            return Err(format!(
                "{field}.data_type must be 'float32_le', got '{}'",
                self.data_type
            ));
        }
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(format!(
                "{field}.shape must contain only positive dimensions"
            ));
        }
        let element_count = self.shape.iter().try_fold(1usize, |product, dim| {
            product
                .checked_mul(*dim as usize)
                .ok_or_else(|| format!("{field}.shape overflows usize"))
        })?;
        if self.data.is_some() == self.reference.is_some() {
            return Err(format!(
                "{field} must contain exactly one of data or reference"
            ));
        }
        let bytes = if let Some(data) = &self.data {
            BASE64
                .decode(data)
                .map_err(|error| format!("{field}.data is not valid base64: {error}"))?
        } else {
            let reference = self.reference.as_ref().unwrap();
            if reference.data_type != self.data_type || reference.shape != self.shape {
                return Err(format!(
                    "{field}.reference data_type/shape does not match the artifact envelope"
                ));
            }
            if reference.runtime_version != NATIVE_RUNTIME_VERSION {
                return Err(format!(
                    "{field}.reference runtime_version mismatch: got '{}', expected '{}'",
                    reference.runtime_version, NATIVE_RUNTIME_VERSION
                ));
            }
            validate_non_empty(&reference.uri, &format!("{field}.reference.uri"))?;
            validate_checksum_syntax(&reference.checksum, &format!("{field}.reference.checksum"))?;
            let bytes = read_artifact_uri(&reference.uri, field).await?;
            if bytes.len() as u64 != reference.byte_length {
                return Err(format!(
                    "{field}.reference byte length mismatch: got {}, expected {}",
                    bytes.len(),
                    reference.byte_length
                ));
            }
            let actual_checksum = sha256_checksum(&bytes);
            if actual_checksum != reference.checksum {
                return Err(format!(
                    "{field}.reference checksum mismatch: got {actual_checksum}, expected {}",
                    reference.checksum
                ));
            }
            bytes
        };
        let expected_bytes = element_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("{field} byte length overflows usize"))?;
        if bytes.len() != expected_bytes {
            return Err(format!(
                "{field}.data has {} bytes, expected {expected_bytes} from shape {:?}",
                bytes.len(),
                self.shape
            ));
        }
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{field}.data contains non-finite values"));
        }
        Ok(LoadedFloat32Artifact { bytes, values })
    }
}

struct LoadedFloat32Artifact {
    bytes: Vec<u8>,
    values: Vec<f32>,
}

fn sha256_checksum(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn validate_checksum_syntax(checksum: &str, field: &str) -> Result<(), String> {
    let digest = checksum
        .strip_prefix("sha256:")
        .ok_or_else(|| format!("{field} must use the sha256:<hex> form"))?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{field} must contain a 64-character SHA-256 hex digest"
        ));
    }
    Ok(())
}

async fn read_artifact_uri(uri: &str, field: &str) -> Result<Vec<u8>, String> {
    match Url::parse(uri) {
        Ok(url) => {
            let (store, location) = parse_url_opts(&url, std::env::vars()).map_err(|error| {
                format!("failed to resolve {field}.reference.uri '{uri}': {error}")
            })?;
            store
                .get(&location)
                .await
                .map_err(|error| format!("failed to read {field}.reference.uri '{uri}': {error}"))?
                .bytes()
                .await
                .map(|bytes| bytes.to_vec())
                .map_err(|error| {
                    format!("failed to download {field}.reference.uri '{uri}': {error}")
                })
        }
        Err(_) => tokio::fs::read(uri)
            .await
            .map_err(|error| format!("failed to read {field}.reference.uri '{uri}': {error}")),
    }
}

fn safe_artifact_name(model_id: &str) -> String {
    let name = model_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if name.is_empty() {
        "model".to_string()
    } else {
        name
    }
}

fn artifact_uri(output_uri: &str, file_name: &str) -> Result<String, String> {
    if let Ok(mut url) = Url::parse(output_uri) {
        let mut path = url.path().trim_end_matches('/').to_string();
        path.push('/');
        path.push_str(file_name);
        url.set_path(&path);
        Ok(url.to_string())
    } else {
        let output = Path::new(output_uri);
        if output.as_os_str().is_empty() {
            return Err("artifact_output_uri must not be empty".to_string());
        }
        Ok(output.join(file_name).to_string_lossy().into_owned())
    }
}

async fn publish_artifact(
    output_uri: &str,
    model_id: &str,
    kind: &str,
    shape: &[u32],
    bytes: &[u8],
) -> Result<ModelArtifactReference, String> {
    let checksum = sha256_checksum(bytes);
    let digest = checksum.strip_prefix("sha256:").unwrap();
    let file_name = format!(
        "{}.{}.{}.f32le",
        safe_artifact_name(model_id),
        kind,
        &digest[..16]
    );
    let uri = artifact_uri(output_uri, &file_name)?;
    let sequence = ARTIFACT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp_suffix = format!(".tmp.{}.{}", std::process::id(), sequence);

    if let Ok(url) = Url::parse(&uri) {
        let (store, location) = parse_url_opts(&url, std::env::vars())
            .map_err(|error| format!("failed to resolve model artifact URI '{uri}': {error}"))?;
        let temp = ObjectStorePath::parse(format!("{location}{temp_suffix}"))
            .map_err(|error| format!("failed to construct temporary artifact URI: {error}"))?;
        store
            .put(&temp, bytes.to_vec().into())
            .await
            .map_err(|error| {
                format!("failed to upload temporary model artifact '{uri}': {error}")
            })?;
        if let Err(error) = store.rename(&temp, &location).await {
            let _ = store.delete(&temp).await;
            return Err(format!("failed to publish model artifact '{uri}': {error}"));
        }
    } else {
        let path = PathBuf::from(&uri);
        let parent = path
            .parent()
            .ok_or_else(|| format!("model artifact path '{uri}' has no parent directory"))?;
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| format!("failed to create model artifact directory: {error}"))?;
        let temp = PathBuf::from(format!("{}{temp_suffix}", path.to_string_lossy()));
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .await
            .map_err(|error| format!("failed to create temporary model artifact: {error}"))?;
        if let Err(error) = file.write_all(bytes).await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(format!("failed to write temporary model artifact: {error}"));
        }
        if let Err(error) = file.sync_all().await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(format!("failed to sync temporary model artifact: {error}"));
        }
        drop(file);
        if let Err(error) = tokio::fs::rename(&temp, &path).await {
            let _ = tokio::fs::remove_file(&temp).await;
            return Err(format!("failed to publish model artifact '{uri}': {error}"));
        }
    }

    Ok(ModelArtifactReference {
        uri,
        data_type: "float32_le".to_string(),
        shape: shape.to_vec(),
        checksum,
        byte_length: bytes.len() as u64,
        runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
    })
}

fn inline_artifact(shape: Vec<u32>, bytes: Vec<u8>) -> Float32Artifact {
    Float32Artifact {
        data_type: "float32_le".to_string(),
        shape,
        data: Some(BASE64.encode(bytes)),
        reference: None,
    }
}

fn referenced_artifact(shape: Vec<u32>, reference: ModelArtifactReference) -> Float32Artifact {
    Float32Artifact {
        data_type: "float32_le".to_string(),
        shape,
        data: None,
        reference: Some(reference),
    }
}

fn update_checksum_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn canonical_model_checksum(
    config: &NativeIndexConfig,
    config_digest: &str,
    model_id: &str,
    model_scope: &str,
    centroids: &LoadedFloat32Artifact,
    pq_codebook: Option<&LoadedFloat32Artifact>,
) -> Result<String, String> {
    let config_json = serde_json::to_vec(config)
        .map_err(|error| format!("failed to canonicalize index_config: {error}"))?;
    let mut hasher = Sha256::new();
    update_checksum_field(&mut hasher, b"lancedb-go-native-index-model-v1");
    update_checksum_field(&mut hasher, NATIVE_RUNTIME_VERSION.as_bytes());
    update_checksum_field(&mut hasher, config_digest.as_bytes());
    update_checksum_field(&mut hasher, &config_json);
    update_checksum_field(&mut hasher, model_id.as_bytes());
    update_checksum_field(&mut hasher, model_scope.as_bytes());
    update_checksum_field(&mut hasher, b"float32_le");
    update_checksum_field(&mut hasher, &centroids.bytes);
    if let Some(codebook) = pq_codebook {
        update_checksum_field(&mut hasher, b"pq_codebook");
        update_checksum_field(&mut hasher, &codebook.bytes);
    } else {
        update_checksum_field(&mut hasher, b"no_pq_codebook");
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn parse_distance_type(distance_type: &str) -> Result<DistanceType, String> {
    match distance_type.to_ascii_lowercase().as_str() {
        "l2" | "euclidean" => Ok(DistanceType::L2),
        "cosine" => Ok(DistanceType::Cosine),
        "dot" => Ok(DistanceType::Dot),
        other => Err(format!("unsupported distance_type '{other}'")),
    }
}

fn validate_index_config(config: &NativeIndexConfig) -> Result<(), String> {
    if config.dimension == 0 || config.num_partitions == 0 {
        return Err("index_config.dimension and num_partitions must be positive".to_string());
    }
    parse_distance_type(&config.distance_type)?;
    match config.index_type.to_ascii_lowercase().as_str() {
        "ivf_flat" => {
            if config.num_sub_vectors.is_some() || config.num_bits.is_some() {
                return Err(
                    "IVF_FLAT must not set index_config.num_sub_vectors or num_bits".to_string(),
                );
            }
        }
        "ivf_pq" => {
            let num_sub_vectors = config
                .num_sub_vectors
                .ok_or_else(|| "IVF_PQ requires index_config.num_sub_vectors".to_string())?;
            let num_bits = config
                .num_bits
                .ok_or_else(|| "IVF_PQ requires index_config.num_bits".to_string())?;
            if num_sub_vectors == 0
                || !config.dimension.is_multiple_of(num_sub_vectors)
            {
                return Err(format!(
                    "IVF_PQ num_sub_vectors {num_sub_vectors} must divide dimension {}",
                    config.dimension
                ));
            }
            if !matches!(num_bits, 4 | 8) {
                return Err("IVF_PQ num_bits must be 4 or 8".to_string());
            }
        }
        other => {
            return Err(format!(
                "unsupported native segment index type '{other}'; supported: IVF_FLAT, IVF_PQ"
            ));
        }
    }
    Ok(())
}

async fn vector_params(request: &SegmentBuildRequest) -> Result<VectorIndexParams, String> {
    let config = &request.index_config;
    validate_index_config(config)?;
    validate_model_identity(&request.model.identity)?;

    let centroids = request.model.centroids.load("model.centroids").await?;
    let expected_centroid_shape = vec![config.num_partitions, config.dimension];
    if request.model.centroids.shape != expected_centroid_shape {
        return Err(format!(
            "model.centroids.shape must be {:?}, got {:?}",
            expected_centroid_shape, request.model.centroids.shape
        ));
    }
    let pq_codebook = if let Some(codebook) = request.model.pq_codebook.as_ref() {
        Some(codebook.load("model.pq_codebook").await?)
    } else {
        None
    };
    let expected_checksum = canonical_model_checksum(
        config,
        &request.index_config_digest,
        &request.model.identity.model_id,
        &request.model.identity.model_scope,
        &centroids,
        pq_codebook.as_ref(),
    )?;
    if request.model.identity.model_checksum != expected_checksum {
        return Err(format!(
            "model checksum mismatch: got '{}', Rust computed '{}'",
            request.model.identity.model_checksum, expected_checksum
        ));
    }

    let centroid_values = Float32Array::from(centroids.values);
    let centroids = FixedSizeListArray::try_new(
        Arc::new(arrow_schema::Field::new(
            "item",
            arrow_schema::DataType::Float32,
            true,
        )),
        i32::try_from(config.dimension)
            .map_err(|_| "index_config.dimension does not fit i32".to_string())?,
        Arc::new(centroid_values),
        None,
    )
    .map_err(|error| format!("invalid IVF centroids: {error}"))?;
    let mut ivf =
        IvfBuildParams::try_with_centroids(config.num_partitions as usize, Arc::new(centroids))
            .map_err(|error| format!("invalid IVF parameters: {error}"))?;
    if let Some(value) = config.max_iterations {
        ivf.max_iters = value as usize;
    }
    if let Some(value) = config.sample_rate {
        ivf.sample_rate = value as usize;
    }
    if let Some(value) = config.target_partition_size {
        ivf.target_partition_size = Some(value as usize);
    }

    let metric = parse_distance_type(&config.distance_type)?;
    match config.index_type.to_ascii_lowercase().as_str() {
        "ivf_flat" => {
            if pq_codebook.is_some() {
                return Err("IVF_FLAT must not provide model.pq_codebook".to_string());
            }
            Ok(VectorIndexParams::with_ivf_flat_params(metric, ivf))
        }
        "ivf_pq" => {
            let num_sub_vectors = config
                .num_sub_vectors
                .ok_or_else(|| "IVF_PQ requires index_config.num_sub_vectors".to_string())?;
            let num_bits = config
                .num_bits
                .ok_or_else(|| "IVF_PQ requires index_config.num_bits".to_string())?;
            if num_sub_vectors == 0
                || !config.dimension.is_multiple_of(num_sub_vectors)
            {
                return Err(format!(
                    "IVF_PQ num_sub_vectors {num_sub_vectors} must divide dimension {}",
                    config.dimension
                ));
            }
            if !matches!(num_bits, 4 | 8) {
                return Err("IVF_PQ num_bits must be 4 or 8".to_string());
            }
            let codebook =
                request.model.pq_codebook.as_ref().ok_or_else(|| {
                    "IVF_PQ distributed build requires model.pq_codebook".to_string()
                })?;
            let expected_codebook_shape = vec![
                num_sub_vectors,
                1u32.checked_shl(num_bits)
                    .ok_or_else(|| "IVF_PQ num_bits overflows codebook shape".to_string())?,
                config.dimension / num_sub_vectors,
            ];
            if codebook.shape != expected_codebook_shape {
                return Err(format!(
                    "model.pq_codebook.shape must be {:?}, got {:?}",
                    expected_codebook_shape, codebook.shape
                ));
            }
            let codebook: ArrayRef = Arc::new(Float32Array::from(pq_codebook.unwrap().values));
            let mut pq =
                PQBuildParams::with_codebook(num_sub_vectors as usize, num_bits as usize, codebook);
            if let Some(value) = config.max_iterations {
                pq.max_iters = value as usize;
            }
            if let Some(value) = config.sample_rate {
                pq.sample_rate = value as usize;
            }
            Ok(VectorIndexParams::with_ivf_pq_params(metric, ivf, pq))
        }
        other => unreachable!("validated index type {other}"),
    }
}

async fn dataset_at_version(table: &lancedb::Table, version: u64) -> Result<Dataset, String> {
    let wrapper = table
        .dataset()
        .ok_or_else(|| "Native Index Segment APIs require a native LanceDB table".to_string())?;
    let current = wrapper
        .get()
        .await
        .map_err(|error| format!("failed to get native dataset: {error}"))?;
    if current.version().version == version {
        Ok(current.as_ref().clone())
    } else {
        current
            .checkout_version(version)
            .await
            .map_err(|error| format!("failed to open dataset version {version}: {error}"))
    }
}

fn dataset_fragment_ids(dataset: &Dataset) -> Result<BTreeSet<u32>, String> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| {
            u32::try_from(fragment.id()).map_err(|_| {
                format!(
                    "fragment id {} exceeds Lance distributed-index u32 range",
                    fragment.id()
                )
            })
        })
        .collect()
}

fn validate_fragment_ids(dataset: &Dataset, fragment_ids: &[u32]) -> Result<(), String> {
    if fragment_ids.is_empty() {
        return Err("fragment_ids must not be empty".to_string());
    }
    let requested = fragment_ids.iter().copied().collect::<BTreeSet<_>>();
    if requested.len() != fragment_ids.len() {
        return Err("fragment_ids contains duplicate IDs".to_string());
    }
    let existing = dataset_fragment_ids(dataset)?;
    let invalid = requested.difference(&existing).copied().collect::<Vec<_>>();
    if !invalid.is_empty() {
        return Err(format!(
            "fragment_ids contains IDs not present in dataset version {}: {:?}",
            dataset.version().version,
            invalid
        ));
    }
    Ok(())
}

fn validate_training_column(dataset: &Dataset, column: &str, dimension: u32) -> Result<(), String> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        format!(
            "vector_column '{column}' is not present in dataset version {}",
            dataset.version().version
        )
    })?;
    match field.data_type() {
        DataType::FixedSizeList(child, actual_dimension)
            if actual_dimension == dimension as i32 && child.data_type() == &DataType::Float32 =>
        {
            Ok(())
        }
        actual => Err(format!(
            "vector_column '{column}' must be FixedSizeList<float32, {dimension}>, got {actual}"
        )),
    }
}

fn f32_array_bytes(
    array: &FixedSizeListArray,
    expected_elements: usize,
    field: &str,
) -> Result<Vec<u8>, String> {
    let values = array
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| format!("{field} training output is not float32"))?;
    if values.len() != expected_elements {
        return Err(format!(
            "{field} training output has {} values, expected {expected_elements}",
            values.len()
        ));
    }
    if values.null_count() != 0 {
        return Err(format!("{field} training output contains null values"));
    }
    if values.values().iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{field} training output contains non-finite values"
        ));
    }
    Ok(values
        .values()
        .iter()
        .copied()
        .flat_map(f32::to_le_bytes)
        .collect())
}

async fn prepare_index_model(
    table: &lancedb::Table,
    request: PrepareIndexModelRequest,
) -> Result<PreparedIndexModel, String> {
    validate_wire_version(request.wire_version)?;
    validate_non_empty(&request.vector_column, "vector_column")?;
    validate_non_empty(&request.logical_index_name, "logical_index_name")?;
    validate_non_empty(&request.index_config_digest, "index_config_digest")?;
    validate_non_empty(&request.model_id, "model_id")?;
    validate_non_empty(&request.model_scope, "model_scope")?;
    validate_index_config(&request.index_config)?;
    if let Some(output_uri) = request.artifact_output_uri.as_deref() {
        validate_non_empty(output_uri, "artifact_output_uri")?;
    }
    if request.index_config.dimension >= 4096 && request.artifact_output_uri.is_none() {
        return Err(
            "models with dimension >= 4096 require artifact_output_uri; inline JSON/base64 is reserved for small models"
                .to_string(),
        );
    }

    let dataset = dataset_at_version(table, request.dataset_version).await?;
    validate_fragment_ids(&dataset, &request.training_fragment_ids)?;
    validate_training_column(
        &dataset,
        &request.vector_column,
        request.index_config.dimension,
    )?;

    let config = &request.index_config;
    let metric = parse_distance_type(&config.distance_type)?;
    let mut ivf_params = IvfBuildParams::new(config.num_partitions as usize);
    if let Some(value) = config.max_iterations {
        ivf_params.max_iters = value as usize;
    }
    if let Some(value) = config.sample_rate {
        ivf_params.sample_rate = value as usize;
    }
    if let Some(value) = config.target_partition_size {
        ivf_params.target_partition_size = Some(value as usize);
    }
    let ivf_model = build_ivf_model(
        &dataset,
        &request.vector_column,
        config.dimension as usize,
        metric,
        &ivf_params,
        Some(&request.training_fragment_ids),
        lance_index::progress::noop_progress(),
    )
    .await
    .map_err(|error| format!("build_ivf_model failed: {error}"))?;
    let centroids = ivf_model
        .centroids
        .as_ref()
        .ok_or_else(|| "build_ivf_model returned no centroids".to_string())?;
    let centroid_shape = vec![config.num_partitions, config.dimension];
    let centroid_bytes = f32_array_bytes(
        centroids,
        (config.num_partitions as usize)
            .checked_mul(config.dimension as usize)
            .ok_or_else(|| "centroid element count overflows usize".to_string())?,
        "centroids",
    )?;

    let (pq_shape, pq_bytes) = if config.index_type.eq_ignore_ascii_case("ivf_pq") {
        let num_sub_vectors = config.num_sub_vectors.unwrap();
        let num_bits = config.num_bits.unwrap();
        let mut pq_params = PQBuildParams::new(num_sub_vectors as usize, num_bits as usize);
        if let Some(value) = config.max_iterations {
            pq_params.max_iters = value as usize;
        }
        if let Some(value) = config.sample_rate {
            pq_params.sample_rate = value as usize;
        }
        let pq_model = build_pq_model_in_fragments(
            &dataset,
            &request.vector_column,
            config.dimension as usize,
            metric,
            &pq_params,
            Some(&ivf_model),
            Some(&request.training_fragment_ids),
        )
        .await
        .map_err(|error| format!("build_pq_model_in_fragments failed: {error}"))?;
        let num_codes = 1u32
            .checked_shl(num_bits)
            .ok_or_else(|| "PQ num_bits overflows codebook shape".to_string())?;
        let shape = vec![
            num_sub_vectors,
            num_codes,
            config.dimension / num_sub_vectors,
        ];
        let expected_elements = (num_codes as usize)
            .checked_mul(config.dimension as usize)
            .ok_or_else(|| "PQ codebook element count overflows usize".to_string())?;
        let bytes = f32_array_bytes(&pq_model.codebook, expected_elements, "pq_codebook")?;
        (Some(shape), Some(bytes))
    } else {
        (None, None)
    };

    let loaded_centroids = LoadedFloat32Artifact {
        values: Vec::new(),
        bytes: centroid_bytes.clone(),
    };
    let loaded_pq = pq_bytes.as_ref().map(|bytes| LoadedFloat32Artifact {
        values: Vec::new(),
        bytes: bytes.clone(),
    });
    let model_checksum = canonical_model_checksum(
        config,
        &request.index_config_digest,
        &request.model_id,
        &request.model_scope,
        &loaded_centroids,
        loaded_pq.as_ref(),
    )?;

    let total_bytes = centroid_bytes
        .len()
        .checked_add(pq_bytes.as_ref().map_or(0, Vec::len))
        .ok_or_else(|| "model artifact byte length overflows usize".to_string())?;
    let (centroid_artifact, pq_artifact) = if let Some(output_uri) =
        request.artifact_output_uri.as_deref()
    {
        let centroid_reference = publish_artifact(
            output_uri,
            &request.model_id,
            "centroids",
            &centroid_shape,
            &centroid_bytes,
        )
        .await?;
        let centroids = referenced_artifact(centroid_shape, centroid_reference);
        let pq = match (pq_shape, pq_bytes) {
            (Some(shape), Some(bytes)) => {
                let reference =
                    publish_artifact(output_uri, &request.model_id, "pq-codebook", &shape, &bytes)
                        .await?;
                Some(referenced_artifact(shape, reference))
            }
            _ => None,
        };
        (centroids, pq)
    } else {
        let inline_limit = request
            .inline_artifact_max_bytes
            .unwrap_or(DEFAULT_INLINE_ARTIFACT_MAX_BYTES as u64);
        if total_bytes as u64 > inline_limit {
            return Err(format!(
                "trained model is {total_bytes} bytes, exceeding inline_artifact_max_bytes={inline_limit}; set artifact_output_uri for Rust-side publication"
            ));
        }
        (
            inline_artifact(centroid_shape, centroid_bytes),
            pq_shape
                .zip(pq_bytes)
                .map(|(shape, bytes)| inline_artifact(shape, bytes)),
        )
    };

    Ok(PreparedIndexModel {
        wire_version: NATIVE_SEGMENT_WIRE_VERSION,
        runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
        dataset_version: request.dataset_version,
        training_fragment_ids: request.training_fragment_ids,
        vector_column: request.vector_column,
        logical_index_name: request.logical_index_name,
        index_config_digest: request.index_config_digest,
        index_config: request.index_config,
        model: IndexModel {
            identity: ModelIdentity {
                model_id: request.model_id,
                model_checksum,
                model_scope: request.model_scope,
                runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
            },
            centroids: centroid_artifact,
            pq_codebook: pq_artifact,
        },
    })
}

fn fragment_ids(metadata: &IndexMetadata) -> Result<Vec<u32>, String> {
    metadata
        .fragment_bitmap
        .as_ref()
        .map(|bitmap| bitmap.iter().collect())
        .ok_or_else(|| format!("segment {} is missing fragment coverage", metadata.uuid))
}

fn metadata_to_envelope(
    metadata: &IndexMetadata,
    vector_column: String,
    index_config_digest: Option<String>,
    index_config: Option<NativeIndexConfig>,
    model_identity: Option<ModelIdentity>,
) -> Result<IndexSegmentEnvelope, String> {
    let details = metadata
        .index_details
        .as_ref()
        .ok_or_else(|| format!("segment {} is missing index details", metadata.uuid))?;
    let proto: pb::IndexMetadata = metadata.into();
    Ok(IndexSegmentEnvelope {
        wire_version: NATIVE_SEGMENT_WIRE_VERSION,
        runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
        uuid: metadata.uuid.to_string(),
        logical_index_name: metadata.name.clone(),
        dataset_version: metadata.dataset_version,
        fragment_ids: fragment_ids(metadata)?,
        vector_column,
        index_version: metadata.index_version,
        index_details: IndexDetailsEnvelope {
            type_url: details.type_url.clone(),
            value: BASE64.encode(&details.value),
        },
        opaque_metadata: BASE64.encode(proto.encode_to_vec()),
        index_config_digest,
        index_config,
        model_identity,
    })
}

fn envelope_to_metadata(envelope: &IndexSegmentEnvelope) -> Result<IndexMetadata, String> {
    validate_wire_version(envelope.wire_version)?;
    if envelope.runtime_version != NATIVE_RUNTIME_VERSION {
        return Err(format!(
            "segment {} runtime_version mismatch: got '{}', expected '{}'",
            envelope.uuid, envelope.runtime_version, NATIVE_RUNTIME_VERSION
        ));
    }
    let bytes = BASE64.decode(&envelope.opaque_metadata).map_err(|error| {
        format!(
            "segment {} opaque_metadata is invalid base64: {error}",
            envelope.uuid
        )
    })?;
    let proto = pb::IndexMetadata::decode(bytes.as_slice()).map_err(|error| {
        format!(
            "segment {} opaque_metadata is invalid protobuf: {error}",
            envelope.uuid
        )
    })?;
    let metadata = IndexMetadata::try_from(proto).map_err(|error| {
        format!(
            "segment {} opaque_metadata is invalid: {error}",
            envelope.uuid
        )
    })?;
    let actual_fragment_ids = fragment_ids(&metadata)?;
    let details = metadata
        .index_details
        .as_ref()
        .ok_or_else(|| format!("segment {} is missing index details", envelope.uuid))?;
    let expected_details = BASE64
        .decode(&envelope.index_details.value)
        .map_err(|error| {
            format!(
                "segment {} index_details.value is invalid base64: {error}",
                envelope.uuid
            )
        })?;
    if metadata.uuid.to_string() != envelope.uuid
        || metadata.name != envelope.logical_index_name
        || metadata.dataset_version != envelope.dataset_version
        || actual_fragment_ids != envelope.fragment_ids
        || metadata.index_version != envelope.index_version
        || details.type_url != envelope.index_details.type_url
        || details.value.as_slice() != expected_details.as_slice()
    {
        return Err(format!(
            "segment {} stable envelope fields do not match opaque_metadata",
            envelope.uuid
        ));
    }
    Ok(metadata)
}

fn validate_coverage(
    expected_fragment_ids: &[u32],
    segments: &[IndexSegmentEnvelope],
) -> Result<(), String> {
    if segments.is_empty() {
        return Err("segments must not be empty".to_string());
    }
    let expected = expected_fragment_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if expected.len() != expected_fragment_ids.len() {
        return Err("fragment_ids contains duplicate IDs".to_string());
    }
    let mut actual = BTreeSet::new();
    for segment in segments {
        let local = segment.fragment_ids.iter().copied().collect::<HashSet<_>>();
        if local.len() != segment.fragment_ids.len() {
            return Err(format!(
                "segment {} contains duplicate fragment coverage",
                segment.uuid
            ));
        }
        for fragment_id in &segment.fragment_ids {
            if !actual.insert(*fragment_id) {
                return Err(format!(
                    "fragment coverage overlaps at fragment {fragment_id}"
                ));
            }
        }
    }
    let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
    let unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
    if !missing.is_empty() || !unexpected.is_empty() {
        return Err(format!(
            "fragment coverage mismatch: missing={missing:?}, unexpected={unexpected:?}"
        ));
    }
    Ok(())
}

fn validate_segment_context(
    dataset_version: u64,
    vector_column: &str,
    logical_index_name: &str,
    index_config_digest: &str,
    index_config: &NativeIndexConfig,
    segment: &IndexSegmentEnvelope,
) -> Result<(), String> {
    if segment.dataset_version != dataset_version {
        return Err(format!(
            "segment {} dataset_version mismatch: got {}, expected {}",
            segment.uuid, segment.dataset_version, dataset_version
        ));
    }
    if segment.vector_column != vector_column {
        return Err(format!(
            "segment {} vector_column mismatch: got '{}', expected '{}'",
            segment.uuid, segment.vector_column, vector_column
        ));
    }
    if segment.logical_index_name != logical_index_name {
        return Err(format!(
            "segment {} logical_index_name mismatch: got '{}', expected '{}'",
            segment.uuid, segment.logical_index_name, logical_index_name
        ));
    }
    if segment.index_config_digest.as_deref() != Some(index_config_digest) {
        return Err(format!(
            "segment {} index_config_digest mismatch",
            segment.uuid
        ));
    }
    if segment.index_config.as_ref() != Some(index_config) {
        return Err(format!("segment {} index_config mismatch", segment.uuid));
    }
    Ok(())
}

async fn list_fragments(
    table: &lancedb::Table,
    dataset_version: u64,
) -> Result<FragmentListResponse, String> {
    let dataset = dataset_at_version(table, dataset_version).await?;
    let fragments = dataset
        .get_fragments()
        .iter()
        .map(|fragment| {
            let metadata = fragment.metadata();
            let physical_row_count = metadata.physical_rows.unwrap_or_default() as u64;
            let row_count = metadata.num_rows().unwrap_or_default() as u64;
            let deletion_rate = if physical_row_count == 0 {
                0.0
            } else {
                (physical_row_count.saturating_sub(row_count)) as f64 / physical_row_count as f64
            };
            let data_bytes = metadata
                .files
                .iter()
                .filter_map(|file| file.file_size_bytes.get())
                .map(|size| size.get())
                .sum();
            FragmentInfo {
                id: metadata.id,
                row_count,
                physical_row_count,
                data_bytes,
                deletion_rate,
            }
        })
        .collect();
    Ok(FragmentListResponse {
        wire_version: NATIVE_SEGMENT_WIRE_VERSION,
        runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
        dataset_version,
        fragments,
    })
}

async fn create_index_uncommitted(
    table: &lancedb::Table,
    request: SegmentBuildRequest,
) -> Result<IndexSegmentEnvelope, String> {
    validate_wire_version(request.wire_version)?;
    validate_non_empty(&request.vector_column, "vector_column")?;
    validate_non_empty(&request.logical_index_name, "logical_index_name")?;
    validate_non_empty(&request.index_config_digest, "index_config_digest")?;
    let params = vector_params(&request).await?;
    let mut dataset = dataset_at_version(table, request.dataset_version).await?;
    validate_fragment_ids(&dataset, &request.fragment_ids)?;

    let columns = [request.vector_column.as_str()];
    let mut builder = dataset.create_index_builder(&columns, LanceIndexType::Vector, &params);
    builder = builder
        .name(request.logical_index_name.clone())
        // A rebuild is staged beside the currently committed logical index.
        // execute_uncommitted does not publish or remove that old generation;
        // commit_existing_index_segments performs the atomic replacement.
        .replace(true)
        .fragments(request.fragment_ids.clone());
    let metadata = builder
        .execute_uncommitted()
        .await
        .map_err(|error| format!("execute_uncommitted failed: {error}"))?;
    let envelope = metadata_to_envelope(
        &metadata,
        request.vector_column,
        Some(request.index_config_digest),
        Some(request.index_config),
        Some(request.model.identity),
    )?;
    let expected = request
        .fragment_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let actual = envelope
        .fragment_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(format!(
            "execute_uncommitted returned unexpected fragment coverage: expected={expected:?}, actual={actual:?}"
        ));
    }
    Ok(envelope)
}

async fn merge_existing_index_segments(
    table: &lancedb::Table,
    request: MergeSegmentsRequest,
) -> Result<IndexSegmentEnvelope, String> {
    validate_wire_version(request.wire_version)?;
    validate_non_empty(&request.vector_column, "vector_column")?;
    validate_non_empty(&request.logical_index_name, "logical_index_name")?;
    validate_non_empty(&request.index_config_digest, "index_config_digest")?;
    validate_model_identity(&request.model_identity)?;
    let dataset = dataset_at_version(table, request.dataset_version).await?;
    validate_fragment_ids(&dataset, &request.fragment_ids)?;
    validate_coverage(&request.fragment_ids, &request.segments)?;

    let mut metadata = Vec::with_capacity(request.segments.len());
    for segment in &request.segments {
        validate_segment_context(
            request.dataset_version,
            &request.vector_column,
            &request.logical_index_name,
            &request.index_config_digest,
            &request.index_config,
            segment,
        )?;
        if segment.model_identity.as_ref() != Some(&request.model_identity) {
            return Err(format!(
                "segment {} model identity is incompatible with physical merge",
                segment.uuid
            ));
        }
        metadata.push(envelope_to_metadata(segment)?);
    }

    let merged = dataset
        .merge_existing_index_segments(metadata)
        .await
        .map_err(|error| format!("merge_existing_index_segments failed: {error}"))?;
    let envelope = metadata_to_envelope(
        &merged,
        request.vector_column,
        Some(request.index_config_digest),
        Some(request.index_config),
        Some(request.model_identity),
    )?;
    validate_coverage(&request.fragment_ids, std::slice::from_ref(&envelope))?;
    Ok(envelope)
}

async fn commit_existing_index_segments(
    table: &lancedb::Table,
    request: CommitSegmentsRequest,
) -> Result<CommitSegmentsResponse, String> {
    validate_wire_version(request.wire_version)?;
    validate_non_empty(&request.vector_column, "vector_column")?;
    validate_non_empty(&request.logical_index_name, "logical_index_name")?;
    validate_non_empty(&request.index_config_digest, "index_config_digest")?;
    validate_coverage(&request.fragment_ids, &request.segments)?;

    let wrapper = table
        .dataset()
        .ok_or_else(|| "Native Index Segment APIs require a native LanceDB table".to_string())?;
    wrapper
        .ensure_mutable()
        .map_err(|error| format!("table is not mutable: {error}"))?;
    let current = wrapper
        .get()
        .await
        .map_err(|error| format!("failed to get native dataset: {error}"))?;
    let mut dataset = current.as_ref().clone();
    dataset
        .checkout_latest()
        .await
        .map_err(|error| format!("failed to refresh the latest dataset version: {error}"))?;
    if dataset.version().version != request.dataset_version {
        let current_version = dataset.version().version;
        wrapper.update(dataset);
        return Err(format!(
            "dataset version mismatch: current={}, requested={}",
            current_version, request.dataset_version
        ));
    }
    validate_fragment_ids(&dataset, &request.fragment_ids)?;
    let all_fragments = dataset_fragment_ids(&dataset)?;
    let requested_fragments = request
        .fragment_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if requested_fragments != all_fragments {
        let missing = all_fragments
            .difference(&requested_fragments)
            .copied()
            .collect::<Vec<_>>();
        let unexpected = requested_fragments
            .difference(&all_fragments)
            .copied()
            .collect::<Vec<_>>();
        return Err(format!(
            "commit requires complete dataset coverage: missing={missing:?}, unexpected={unexpected:?}"
        ));
    }

    let mut metadata = Vec::with_capacity(request.segments.len());
    let mut segment_uuids = Vec::with_capacity(request.segments.len());
    for segment in &request.segments {
        validate_segment_context(
            request.dataset_version,
            &request.vector_column,
            &request.logical_index_name,
            &request.index_config_digest,
            &request.index_config,
            segment,
        )?;
        let model_identity = segment.model_identity.as_ref().ok_or_else(|| {
            format!(
                "segment {} is missing the build model identity",
                segment.uuid
            )
        })?;
        validate_model_identity(model_identity)?;
        metadata.push(envelope_to_metadata(segment)?);
        segment_uuids.push(segment.uuid.clone());
    }

    dataset
        .commit_existing_index_segments(
            &request.logical_index_name,
            &request.vector_column,
            metadata,
        )
        .await
        .map_err(|error| format!("commit_existing_index_segments failed: {error}"))?;
    let committed_dataset_version = dataset.version().version;
    wrapper.update(dataset);

    Ok(CommitSegmentsResponse {
        wire_version: NATIVE_SEGMENT_WIRE_VERSION,
        runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
        logical_index_name: request.logical_index_name,
        source_dataset_version: request.dataset_version,
        committed_dataset_version,
        vector_column: request.vector_column,
        fragment_ids: request.fragment_ids,
        segment_uuids,
    })
}

async fn inspect_index_segments(
    table: &lancedb::Table,
    index_name: &str,
) -> Result<InspectIndexSegmentsResponse, String> {
    validate_non_empty(index_name, "index_name")?;
    let wrapper = table
        .dataset()
        .ok_or_else(|| "Native Index Segment APIs require a native LanceDB table".to_string())?;
    let dataset = wrapper
        .get()
        .await
        .map_err(|error| format!("failed to get native dataset: {error}"))?;
    let metadata = dataset
        .load_indices_by_name(index_name)
        .await
        .map_err(|error| format!("failed to inspect index '{index_name}': {error}"))?;
    let vector_column = metadata
        .first()
        .and_then(|segment| segment.fields.first())
        .and_then(|field_id| dataset.schema().field_by_id(*field_id))
        .map(|field| field.name.clone());
    let segments = metadata
        .iter()
        .map(|segment| {
            metadata_to_envelope(
                segment,
                vector_column.clone().unwrap_or_default(),
                None,
                None,
                None,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(InspectIndexSegmentsResponse {
        wire_version: NATIVE_SEGMENT_WIRE_VERSION,
        runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
        dataset_version: dataset.version().version,
        logical_index_name: index_name.to_string(),
        vector_column,
        segments,
    })
}

fn parse_json_request<T: for<'de> Deserialize<'de>>(
    data: *const u8,
    len: usize,
) -> Result<T, String> {
    if data.is_null() || len == 0 {
        return Err("request bytes must not be null or empty".to_string());
    }
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    serde_json::from_slice(bytes).map_err(|error| format!("invalid request JSON: {error}"))
}

fn write_json_response<T: Serialize>(
    value: &T,
    output: *mut *mut u8,
    output_len: *mut usize,
) -> Result<(), String> {
    if output.is_null() || output_len.is_null() {
        return Err("output and output_len must not be null".to_string());
    }
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("failed to serialize response JSON: {error}"))?
        .into_boxed_slice();
    let len = bytes.len();
    let data = Box::into_raw(bytes) as *mut u8;
    unsafe {
        *output = data;
        *output_len = len;
    }
    Ok(())
}

fn initialize_output(output: *mut *mut u8, output_len: *mut usize) {
    unsafe {
        if !output.is_null() {
            *output = ptr::null_mut();
        }
        if !output_len.is_null() {
            *output_len = 0;
        }
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown Rust panic".to_string()
    }
}

fn ffi_result<F>(operation: &'static str, function: F) -> *mut SimpleResult
where
    F: FnOnce() -> Result<(), String>,
{
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(())) => Box::into_raw(Box::new(SimpleResult::ok())),
        Ok(Err(error)) => Box::into_raw(Box::new(SimpleResult::error(error))),
        Err(payload) => Box::into_raw(Box::new(SimpleResult::panic(format!(
            "panic in {operation}: {}",
            panic_message(payload)
        )))),
    }
}

/// Enumerate fragments from exactly `dataset_version`.
#[no_mangle]
pub extern "C" fn simple_lancedb_table_list_fragments(
    table_handle: *mut c_void,
    dataset_version: u64,
    output: *mut *mut u8,
    output_len: *mut usize,
) -> *mut SimpleResult {
    initialize_output(output, output_len);
    ffi_result("simple_lancedb_table_list_fragments", || {
        if table_handle.is_null() {
            return Err("table_handle must not be null".to_string());
        }
        let table = unsafe { &*(table_handle as *const lancedb::Table) };
        let response = get_simple_runtime().block_on(list_fragments(table, dataset_version))?;
        write_json_response(&response, output, output_len)
    })
}

/// Train a shared IVF / IVF_PQ model on an exact dataset snapshot and fragment set.
#[no_mangle]
pub extern "C" fn simple_lancedb_table_prepare_index_model(
    table_handle: *mut c_void,
    request: *const u8,
    request_len: usize,
    output: *mut *mut u8,
    output_len: *mut usize,
) -> *mut SimpleResult {
    initialize_output(output, output_len);
    ffi_result("simple_lancedb_table_prepare_index_model", || {
        if table_handle.is_null() {
            return Err("table_handle must not be null".to_string());
        }
        let request = parse_json_request::<PrepareIndexModelRequest>(request, request_len)?;
        let table = unsafe { &*(table_handle as *const lancedb::Table) };
        let response = get_simple_runtime().block_on(prepare_index_model(table, request))?;
        write_json_response(&response, output, output_len)
    })
}

/// Build one fragment-scoped uncommitted vector index segment.
#[no_mangle]
pub extern "C" fn simple_lancedb_table_create_index_uncommitted(
    table_handle: *mut c_void,
    request: *const u8,
    request_len: usize,
    output: *mut *mut u8,
    output_len: *mut usize,
) -> *mut SimpleResult {
    initialize_output(output, output_len);
    ffi_result("simple_lancedb_table_create_index_uncommitted", || {
        if table_handle.is_null() {
            return Err("table_handle must not be null".to_string());
        }
        let request = parse_json_request::<SegmentBuildRequest>(request, request_len)?;
        let table = unsafe { &*(table_handle as *const lancedb::Table) };
        let response = get_simple_runtime().block_on(create_index_uncommitted(table, request))?;
        write_json_response(&response, output, output_len)
    })
}

/// Physically merge compatible uncommitted segments.
#[no_mangle]
pub extern "C" fn simple_lancedb_table_merge_existing_index_segments(
    table_handle: *mut c_void,
    request: *const u8,
    request_len: usize,
    output: *mut *mut u8,
    output_len: *mut usize,
) -> *mut SimpleResult {
    initialize_output(output, output_len);
    ffi_result("simple_lancedb_table_merge_existing_index_segments", || {
        if table_handle.is_null() {
            return Err("table_handle must not be null".to_string());
        }
        let request = parse_json_request::<MergeSegmentsRequest>(request, request_len)?;
        let table = unsafe { &*(table_handle as *const lancedb::Table) };
        let response =
            get_simple_runtime().block_on(merge_existing_index_segments(table, request))?;
        write_json_response(&response, output, output_len)
    })
}

/// Atomically commit physical segments as one logical vector index.
#[no_mangle]
pub extern "C" fn simple_lancedb_table_commit_existing_index_segments(
    table_handle: *mut c_void,
    request: *const u8,
    request_len: usize,
    output: *mut *mut u8,
    output_len: *mut usize,
) -> *mut SimpleResult {
    initialize_output(output, output_len);
    ffi_result(
        "simple_lancedb_table_commit_existing_index_segments",
        || {
            if table_handle.is_null() {
                return Err("table_handle must not be null".to_string());
            }
            let request = parse_json_request::<CommitSegmentsRequest>(request, request_len)?;
            let table = unsafe { &*(table_handle as *const lancedb::Table) };
            let response =
                get_simple_runtime().block_on(commit_existing_index_segments(table, request))?;
            write_json_response(&response, output, output_len)
        },
    )
}

/// Inspect committed physical segments for one logical index name.
#[no_mangle]
pub extern "C" fn simple_lancedb_table_inspect_index_segments(
    table_handle: *mut c_void,
    index_name: *const c_char,
    output: *mut *mut u8,
    output_len: *mut usize,
) -> *mut SimpleResult {
    initialize_output(output, output_len);
    ffi_result("simple_lancedb_table_inspect_index_segments", || {
        if table_handle.is_null() || index_name.is_null() {
            return Err("table_handle and index_name must not be null".to_string());
        }
        let index_name =
            from_c_str(index_name).map_err(|error| format!("invalid index_name: {error}"))?;
        let table = unsafe { &*(table_handle as *const lancedb::Table) };
        let response = get_simple_runtime().block_on(inspect_index_segments(table, &index_name))?;
        write_json_response(&response, output, output_len)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn artifact(shape: &[u32], values: &[f32]) -> Float32Artifact {
        Float32Artifact {
            data_type: "float32_le".to_string(),
            shape: shape.to_vec(),
            data: Some(
                BASE64.encode(
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            ),
            reference: None,
        }
    }

    fn segment(uuid: &str, fragments: &[u32]) -> IndexSegmentEnvelope {
        IndexSegmentEnvelope {
            wire_version: NATIVE_SEGMENT_WIRE_VERSION,
            runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
            uuid: uuid.to_string(),
            logical_index_name: "idx".to_string(),
            dataset_version: 7,
            fragment_ids: fragments.to_vec(),
            vector_column: "vector".to_string(),
            index_version: 3,
            index_details: IndexDetailsEnvelope {
                type_url: "type.googleapis.com/lance.VectorIndexDetails".to_string(),
                value: String::new(),
            },
            opaque_metadata: String::new(),
            index_config_digest: Some("sha256:test".to_string()),
            index_config: None,
            model_identity: None,
        }
    }

    #[test]
    fn coverage_rejects_overlap() {
        let error = validate_coverage(&[1, 2, 3], &[segment("a", &[1, 2]), segment("b", &[2, 3])])
            .unwrap_err();
        assert!(error.contains("overlaps"), "{error}");
    }

    #[test]
    fn coverage_rejects_missing_fragment() {
        let error =
            validate_coverage(&[1, 2, 3], &[segment("a", &[1]), segment("b", &[3])]).unwrap_err();
        assert!(error.contains("missing=[2]"), "{error}");
    }

    #[test]
    fn coverage_accepts_disjoint_complete_segments() {
        validate_coverage(
            &[1, 2, 3, 4],
            &[segment("a", &[1, 3]), segment("b", &[2, 4])],
        )
        .unwrap();
    }

    #[test]
    fn ffi_panics_become_errors() {
        let result = ffi_result("panic-test", || -> Result<(), String> { panic!("boom") });
        let result = unsafe { Box::from_raw(result) };
        assert!(!result.success);
        assert_eq!(result.error_code, crate::ffi::FFI_ERROR_PANIC);
        unsafe {
            if !result.error_message.is_null() {
                let _ = CString::from_raw(result.error_message);
            }
            if !result.runtime_version.is_null() {
                let _ = CString::from_raw(result.runtime_version);
            }
        }
    }

    #[tokio::test]
    async fn ivf_pq_requires_precomputed_codebook() {
        let mut request = SegmentBuildRequest {
            wire_version: NATIVE_SEGMENT_WIRE_VERSION,
            dataset_version: 1,
            fragment_ids: vec![0],
            vector_column: "vector".to_string(),
            logical_index_name: "idx".to_string(),
            index_config_digest: "sha256:test".to_string(),
            index_config: NativeIndexConfig {
                index_type: "IVF_PQ".to_string(),
                distance_type: "l2".to_string(),
                dimension: 4,
                num_partitions: 2,
                num_sub_vectors: Some(2),
                num_bits: Some(4),
                max_iterations: None,
                sample_rate: None,
                target_partition_size: None,
            },
            model: IndexModel {
                identity: ModelIdentity {
                    model_id: "model".to_string(),
                    model_checksum: String::new(),
                    model_scope: "macro".to_string(),
                    runtime_version: NATIVE_RUNTIME_VERSION.to_string(),
                },
                centroids: artifact(&[2, 4], &[0.0; 8]),
                pq_codebook: None,
            },
        };

        let centroids = request
            .model
            .centroids
            .load("model.centroids")
            .await
            .unwrap();
        request.model.identity.model_checksum = canonical_model_checksum(
            &request.index_config,
            &request.index_config_digest,
            &request.model.identity.model_id,
            &request.model.identity.model_scope,
            &centroids,
            None,
        )
        .unwrap();
        let error = vector_params(&request).await.unwrap_err();
        assert!(error.contains("requires model.pq_codebook"), "{error}");

        request.model.pq_codebook = Some(artifact(&[2, 16, 2], &[0.0; 64]));
        let codebook = request
            .model
            .pq_codebook
            .as_ref()
            .unwrap()
            .load("model.pq_codebook")
            .await
            .unwrap();
        request.model.identity.model_checksum = canonical_model_checksum(
            &request.index_config,
            &request.index_config_digest,
            &request.model.identity.model_id,
            &request.model.identity.model_scope,
            &centroids,
            Some(&codebook),
        )
        .unwrap();
        vector_params(&request)
            .await
            .expect("a correctly-shaped shared PQ codebook should be accepted");
    }
}
