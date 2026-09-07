//! Durable artifact registry.
//!
//! Registration has to survive a restart. An in-memory map would need replacing the
//! moment anything depends on an artifact identifier outliving the process, and the
//! identifier is meant to be stable.
//!
//! SQLite rather than a file the runtime writes itself, because a registry has
//! concurrent readers and writers and durable updates, and reinventing those
//! semantics over a text file is a well-known way to lose data.
//!
//! # Schema shape
//!
//! Normalised columns hold only what the runtime itself reasons about. Everything
//! else the container declares goes into a key-value table, so encountering a new
//! metadata key is not a schema migration.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, params};

use crate::artifact::{
    ArtifactFingerprint, ArtifactFormat, ArtifactId, ArtifactIntegrity, ArtifactMetadata,
    ArtifactState, DigestAlgorithm, ModelArtifact, detect_drift, render_value,
};
use crate::gguf::{self, GgufError};

/// Why a registry operation failed.
#[derive(Debug)]
pub enum RegistryError {
    /// The path does not exist.
    NotFound { path: PathBuf },
    /// The path is not a regular file.
    NotAFile { path: PathBuf, reason: String },
    /// The file is not a container this runtime can read.
    NotAnArtifact { path: PathBuf, source: GgufError },
    /// The artifact is not registered.
    UnknownArtifact { id: ArtifactId },
    /// Storage failed.
    Storage(rusqlite::Error),
    /// The filesystem failed.
    Io(std::io::Error),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { path } => write!(f, "no such file: {}", path.display()),
            Self::NotAFile { path, reason } => {
                write!(f, "{} is not a model file: {reason}", path.display())
            }
            Self::NotAnArtifact { path, source } => {
                write!(f, "{} cannot be read: {source}", path.display())
            }
            Self::UnknownArtifact { id } => write!(f, "no artifact registered with id {id}"),
            Self::Storage(err) => write!(f, "artifact registry storage failure: {err}"),
            Self::Io(err) => write!(f, "artifact registry filesystem failure: {err}"),
        }
    }
}

impl std::error::Error for RegistryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotAnArtifact { source, .. } => Some(source),
            Self::Storage(err) => Some(err),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for RegistryError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Storage(err)
    }
}

impl From<std::io::Error> for RegistryError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// The outcome of registering a path.
#[derive(Debug, Clone, PartialEq)]
pub enum Registration {
    /// The artifact was recorded for the first time.
    Registered(ModelArtifact),
    /// The path was already registered, and keeps its original identifier.
    AlreadyRegistered(ModelArtifact),
}

impl Registration {
    /// The artifact, however it was arrived at.
    #[must_use]
    pub fn artifact(&self) -> &ModelArtifact {
        match self {
            Self::Registered(artifact) | Self::AlreadyRegistered(artifact) => artifact,
        }
    }

    /// Whether this call created the record.
    #[must_use]
    pub fn is_new(&self) -> bool {
        matches!(self, Self::Registered(_))
    }
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS artifacts (
    id                  TEXT PRIMARY KEY,
    canonical_path      TEXT NOT NULL UNIQUE,
    format              TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL,
    container_version   INTEGER NOT NULL,
    tensor_count        INTEGER NOT NULL,
    metadata_count      INTEGER NOT NULL,
    architecture        TEXT,
    name                TEXT,
    file_type           INTEGER,
    context_length      INTEGER,
    embedding_length    INTEGER,
    tokenizer_model     TEXT,
    chat_template       TEXT,
    registered_at       INTEGER NOT NULL,
    mtime_at_registration INTEGER,
    integrity_state     TEXT NOT NULL,
    digest_algorithm    TEXT,
    digest              BLOB
);

CREATE TABLE IF NOT EXISTS artifact_metadata (
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (artifact_id, key)
);
";

/// A durable record of known artifacts.
///
/// Registration records where a file is. It never copies, moves, or modifies it:
/// taking ownership of a file is a different operation with different consequences,
/// and conflating the two would surprise anyone who registered a model they wanted
/// left where it was.
pub struct ArtifactRegistry {
    connection: Connection,
}

impl std::fmt::Debug for ArtifactRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArtifactRegistry").finish_non_exhaustive()
    }
}

impl ArtifactRegistry {
    /// Opens or creates a registry at a path.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Storage`] when the database cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let connection = Connection::open(path)?;
        Self::prepare(connection)
    }

    /// Creates a registry that exists only for the life of this process.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Storage`] when the database cannot be created.
    pub fn open_in_memory() -> Result<Self, RegistryError> {
        Self::prepare(Connection::open_in_memory()?)
    }

    fn prepare(connection: Connection) -> Result<Self, RegistryError> {
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.execute_batch(SCHEMA)?;
        Ok(Self { connection })
    }

    /// Records an artifact, validating it first.
    ///
    /// The file is inspected, not trusted: its contents decide whether it is a
    /// model container, and its name is not consulted. Registering never starts a
    /// backend, because knowing what an artifact is must not depend on being able
    /// to execute it.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::NotFound`], [`RegistryError::NotAFile`], or
    /// [`RegistryError::NotAnArtifact`] when the path cannot be registered.
    pub fn register(&mut self, path: impl AsRef<Path>) -> Result<Registration, RegistryError> {
        let path = path.as_ref();

        let metadata = std::fs::symlink_metadata(path).map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                RegistryError::NotFound {
                    path: path.to_path_buf(),
                }
            } else {
                RegistryError::Io(err)
            }
        })?;

        if metadata.is_dir() {
            return Err(RegistryError::NotAFile {
                path: path.to_path_buf(),
                reason: "path is a directory".to_owned(),
            });
        }

        // Canonicalising resolves symbolic links and path aliases, so the same file
        // reached by two different routes registers once rather than twice. It also
        // means the recorded path is the file itself, not a link that could later be
        // repointed at something else.
        let canonical = std::fs::canonicalize(path)?;
        let canonical_text = canonical.to_string_lossy().into_owned();

        if let Some(existing) = self.find_by_path(&canonical_text)? {
            return Ok(Registration::AlreadyRegistered(existing));
        }

        let file = std::fs::File::open(&canonical)?;
        let header = gguf::parse(std::io::BufReader::new(file)).map_err(|source| {
            RegistryError::NotAnArtifact {
                path: canonical.clone(),
                source,
            }
        })?;

        let fingerprint = ArtifactFingerprint::observe(&canonical)?;
        let artifact = ModelArtifact {
            id: ArtifactId::generate()?,
            path: canonical,
            format: ArtifactFormat::Gguf,
            size_bytes: fingerprint.size_bytes,
            metadata: ArtifactMetadata::from_header(&header),
            // Registration reads only the header, so nothing is known about the
            // rest of the contents and the record must not pretend otherwise.
            integrity: ArtifactIntegrity::Unverified,
            fingerprint,
            registered_at: SystemTime::now(),
        };

        self.insert(&artifact, &header)?;
        Ok(Registration::Registered(artifact))
    }

    fn insert(
        &mut self,
        artifact: &ModelArtifact,
        header: &gguf::GgufHeader,
    ) -> Result<(), RegistryError> {
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "INSERT INTO artifacts (
                id, canonical_path, format, size_bytes, container_version, tensor_count,
                metadata_count, architecture, name, file_type, context_length,
                embedding_length, tokenizer_model, chat_template, registered_at,
                mtime_at_registration, integrity_state, digest_algorithm, digest
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)",
            params![
                artifact.id.as_str(),
                artifact.path.to_string_lossy(),
                artifact.format.as_str(),
                to_signed(artifact.size_bytes),
                artifact.metadata.container_version,
                to_signed(artifact.metadata.tensor_count),
                to_signed(artifact.metadata.metadata_count),
                artifact.metadata.architecture,
                artifact.metadata.name,
                artifact.metadata.file_type.map(to_signed),
                artifact.metadata.context_length.map(to_signed),
                artifact.metadata.embedding_length.map(to_signed),
                artifact.metadata.tokenizer_model,
                artifact.metadata.chat_template,
                to_epoch(artifact.registered_at),
                artifact.fingerprint.modified_at.map(to_epoch_nanos),
                "unverified",
                None::<String>,
                None::<Vec<u8>>,
            ],
        )?;

        {
            let mut statement = transaction.prepare(
                "INSERT OR REPLACE INTO artifact_metadata (artifact_id, key, value)
                 VALUES (?1, ?2, ?3)",
            )?;
            for (key, value) in &header.metadata {
                statement.execute(params![artifact.id.as_str(), key, render_value(value)])?;
            }
        }

        transaction.commit()?;
        Ok(())
    }

    /// Looks up an artifact by identifier.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Storage`] when the query fails.
    pub fn get(&self, id: &ArtifactId) -> Result<Option<ModelArtifact>, RegistryError> {
        self.connection
            .query_row(
                &format!("SELECT {COLUMNS} FROM artifacts WHERE id = ?1"),
                params![id.as_str()],
                row_to_artifact,
            )
            .optional()
            .map_err(RegistryError::Storage)
    }

    fn find_by_path(&self, canonical: &str) -> Result<Option<ModelArtifact>, RegistryError> {
        self.connection
            .query_row(
                &format!("SELECT {COLUMNS} FROM artifacts WHERE canonical_path = ?1"),
                params![canonical],
                row_to_artifact,
            )
            .optional()
            .map_err(RegistryError::Storage)
    }

    /// Every registered artifact, most recently registered first.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Storage`] when the query fails.
    pub fn list(&self) -> Result<Vec<ModelArtifact>, RegistryError> {
        let mut statement = self.connection.prepare(&format!(
            "SELECT {COLUMNS} FROM artifacts ORDER BY registered_at DESC"
        ))?;
        let rows = statement.query_map([], row_to_artifact)?;
        let mut artifacts = Vec::new();
        for row in rows {
            artifacts.push(row?);
        }
        Ok(artifacts)
    }

    /// All metadata recorded for an artifact, including keys the runtime itself
    /// does not interpret.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Storage`] when the query fails.
    pub fn metadata(&self, id: &ArtifactId) -> Result<Vec<(String, String)>, RegistryError> {
        let mut statement = self.connection.prepare(
            "SELECT key, value FROM artifact_metadata WHERE artifact_id = ?1 ORDER BY key",
        )?;
        let rows = statement.query_map(params![id.as_str()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }
        Ok(entries)
    }

    /// Whether a registered artifact still matches what was recorded.
    ///
    /// A file that has changed is reported, never silently adopted. Updating the
    /// record here would erase the difference between the artifact that was
    /// inspected and whatever now sits at that path.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::UnknownArtifact`] when the identifier is not known.
    pub fn check(&self, id: &ArtifactId) -> Result<ArtifactState, RegistryError> {
        let artifact = self
            .get(id)?
            .ok_or_else(|| RegistryError::UnknownArtifact { id: id.clone() })?;
        Ok(detect_drift(&artifact))
    }

    /// Reads an artifact in full and records a content digest.
    ///
    /// Deliberately separate from registration and never on its critical path. This
    /// reads every byte, which is trivial for a small container and expensive for a
    /// large one, and registration must not become proportional to file size.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::UnknownArtifact`] when the identifier is not known,
    /// or an error when the file cannot be read.
    pub fn verify(&mut self, id: &ArtifactId) -> Result<ArtifactIntegrity, RegistryError> {
        use sha2::{Digest, Sha256};

        let artifact = self
            .get(id)?
            .ok_or_else(|| RegistryError::UnknownArtifact { id: id.clone() })?;

        let mut file = std::fs::File::open(&artifact.path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let digest = hasher.finalize().to_vec();

        self.connection.execute(
            "UPDATE artifacts SET integrity_state = ?1, digest_algorithm = ?2, digest = ?3
             WHERE id = ?4",
            params![
                "verified",
                DigestAlgorithm::Sha256.as_str(),
                digest.clone(),
                id.as_str()
            ],
        )?;

        Ok(ArtifactIntegrity::Verified {
            algorithm: DigestAlgorithm::Sha256,
            digest,
        })
    }

    /// Removes an artifact record. The file itself is untouched.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Storage`] when the deletion fails.
    pub fn forget(&mut self, id: &ArtifactId) -> Result<bool, RegistryError> {
        let removed = self
            .connection
            .execute("DELETE FROM artifacts WHERE id = ?1", params![id.as_str()])?;
        Ok(removed > 0)
    }
}

const COLUMNS: &str = "id, canonical_path, format, size_bytes, container_version, tensor_count, \
     metadata_count, architecture, name, file_type, context_length, embedding_length, \
     tokenizer_model, chat_template, registered_at, mtime_at_registration, integrity_state, \
     digest_algorithm, digest";

fn row_to_artifact(row: &rusqlite::Row<'_>) -> rusqlite::Result<ModelArtifact> {
    let format_text: String = row.get(2)?;
    let integrity_state: String = row.get(16)?;
    let digest_algorithm: Option<String> = row.get(17)?;
    let digest: Option<Vec<u8>> = row.get(18)?;

    let integrity = match (integrity_state.as_str(), digest_algorithm, digest) {
        ("verified", Some(algorithm), Some(digest)) => {
            match DigestAlgorithm::from_stored(&algorithm) {
                Some(algorithm) => ArtifactIntegrity::Verified { algorithm, digest },
                // An algorithm this build does not know is not a verified state.
                None => ArtifactIntegrity::Unverified,
            }
        }
        _ => ArtifactIntegrity::Unverified,
    };

    let modified_at: Option<i64> = row.get(15)?;

    Ok(ModelArtifact {
        id: ArtifactId::from_stored(row.get::<_, String>(0)?),
        path: PathBuf::from(row.get::<_, String>(1)?),
        format: ArtifactFormat::from_stored(&format_text).unwrap_or(ArtifactFormat::Gguf),
        size_bytes: from_signed(row.get(3)?),
        metadata: ArtifactMetadata {
            container_version: row.get(4)?,
            tensor_count: from_signed(row.get(5)?),
            metadata_count: from_signed(row.get(6)?),
            architecture: row.get(7)?,
            name: row.get(8)?,
            file_type: row.get::<_, Option<i64>>(9)?.map(from_signed),
            context_length: row.get::<_, Option<i64>>(10)?.map(from_signed),
            embedding_length: row.get::<_, Option<i64>>(11)?.map(from_signed),
            tokenizer_model: row.get(12)?,
            chat_template: row.get(13)?,
        },
        integrity,
        fingerprint: ArtifactFingerprint {
            size_bytes: from_signed(row.get(3)?),
            modified_at: modified_at.map(from_epoch_nanos),
        },
        registered_at: from_epoch(row.get(14)?),
    })
}

/// SQLite stores signed 64-bit integers only, so unsigned values convert at the
/// storage boundary rather than being silently reinterpreted.
///
/// Saturating is safe for what is stored here: the largest value is a file size,
/// and `i64::MAX` bytes is eight exabytes.
fn to_signed(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn from_signed(value: i64) -> u64 {
    value.max(0).unsigned_abs()
}

fn to_epoch(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |elapsed| {
        i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX)
    })
}

fn from_epoch(seconds: i64) -> SystemTime {
    UNIX_EPOCH + std::time::Duration::from_secs(seconds.max(0).unsigned_abs())
}

/// Stores a modification time without losing precision.
///
/// Whole seconds are not enough. Filesystems report modification times with
/// sub-second precision, so a value truncated on the way in never equals the value
/// read from the file afterwards, and every artifact would report as changed the
/// instant it was registered. Nanoseconds since the epoch fit in a signed 64-bit
/// integer until well beyond any plausible lifetime of this software.
fn to_epoch_nanos(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH).map_or(0, |elapsed| {
        i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX)
    })
}

fn from_epoch_nanos(nanos: i64) -> SystemTime {
    UNIX_EPOCH + std::time::Duration::from_nanos(nanos.max(0).unsigned_abs())
}
