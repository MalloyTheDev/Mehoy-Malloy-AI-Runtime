//! Artifact registration acceptance tests.
//!
//! Malformed containers are synthesised rather than collected, so every rejection
//! path is exercised deterministically instead of depending on finding a corrupt
//! file. Real containers are used where they are available on the machine, since a
//! synthetic container proves the parser and a real one proves the parser matches
//! reality.

use std::io::Write;
use std::path::{Path, PathBuf};

use mehoy_registry::{
    ArtifactIntegrity, ArtifactRegistry, ArtifactState, GgufError, RegistryError,
};

/// Builds GGUF byte streams, including deliberately malformed ones.
#[derive(Default)]
struct Gguf {
    bytes: Vec<u8>,
}

impl Gguf {
    fn new() -> Self {
        Self {
            bytes: b"GGUF".to_vec(),
        }
    }

    fn raw(mut self, bytes: &[u8]) -> Self {
        self.bytes.extend_from_slice(bytes);
        self
    }

    fn u32(self, value: u32) -> Self {
        self.raw(&value.to_le_bytes())
    }

    fn u64(self, value: u64) -> Self {
        self.raw(&value.to_le_bytes())
    }

    fn string(self, value: &str) -> Self {
        self.u64(value.len() as u64).raw(value.as_bytes())
    }

    fn text_entry(self, key: &str, value: &str) -> Self {
        self.string(key).u32(8).string(value)
    }

    fn number_entry(self, key: &str, value: u32) -> Self {
        self.string(key).u32(4).u32(value)
    }

    fn build(self) -> Vec<u8> {
        self.bytes
    }
}

/// A well-formed container with the fields the runtime reasons about.
fn valid_container() -> Vec<u8> {
    Gguf::new()
        .u32(3)
        .u64(12)
        .u64(6)
        .text_entry("general.architecture", "llama")
        .text_entry("general.name", "tiny test model")
        .number_entry("general.file_type", 15)
        .number_entry("llama.context_length", 8192)
        .number_entry("llama.embedding_length", 576)
        .text_entry("tokenizer.ggml.model", "gpt2")
        .build()
}

fn write_file(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    let mut file = std::fs::File::create(&path).expect("test file is creatable");
    file.write_all(bytes).expect("test file is writable");
    file.sync_all().expect("test file is flushed");
    path
}

fn registry() -> ArtifactRegistry {
    ArtifactRegistry::open_in_memory().expect("registry opens")
}

#[test]
fn a_valid_container_registers_with_its_metadata() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &valid_container());
    let mut registry = registry();

    let registration = registry.register(&path).expect("registers");
    assert!(registration.is_new());
    let artifact = registration.artifact();

    assert_eq!(artifact.metadata.architecture.as_deref(), Some("llama"));
    assert_eq!(artifact.metadata.name.as_deref(), Some("tiny test model"));
    assert_eq!(artifact.metadata.context_length, Some(8192));
    assert_eq!(artifact.metadata.embedding_length, Some(576));
    assert_eq!(artifact.metadata.tokenizer_model.as_deref(), Some("gpt2"));
    assert_eq!(artifact.metadata.file_type, Some(15));
    assert_eq!(artifact.metadata.tensor_count, 12);
    assert_eq!(artifact.size_bytes, valid_container().len() as u64);
}

#[test]
fn a_newly_registered_artifact_is_unverified() {
    // Registration reads only the header, so it must not claim to know the
    // contents. Anything else would be a lie the moment a large file is registered.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &valid_container());
    let mut registry = registry();

    let registration = registry.register(&path).expect("registers");
    assert_eq!(
        registration.artifact().integrity,
        ArtifactIntegrity::Unverified
    );
}

#[test]
fn a_gguf_extension_is_not_evidence_of_anything() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(
        dir.path(),
        "definitely.gguf",
        b"I am a text file, not a model",
    );
    let mut registry = registry();

    match registry.register(&path).expect_err("must be rejected") {
        RegistryError::NotAnArtifact { source, .. } => {
            assert!(
                matches!(source, GgufError::NotGguf { .. }),
                "expected NotGguf, got {source}"
            );
        }
        other => panic!("expected NotAnArtifact, got {other}"),
    }
}

#[test]
fn a_container_without_the_extension_still_registers() {
    // The converse of the previous test: contents decide, not the name.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model-with-no-extension", &valid_container());
    let mut registry = registry();
    assert!(registry.register(&path).expect("registers").is_new());
}

#[test]
fn a_truncated_container_is_rejected() {
    let dir = tempfile::tempdir().expect("temp dir");
    let full = valid_container();
    let mut registry = registry();

    for cut in [2, 6, 10, 20, 40, 60] {
        let path = write_file(dir.path(), &format!("cut{cut}.gguf"), &full[..cut]);
        let err = registry.register(&path).expect_err("must be rejected");
        assert!(
            matches!(err, RegistryError::NotAnArtifact { .. }),
            "cutting at {cut} gave {err}"
        );
    }
}

#[test]
fn an_unsupported_version_is_reported_explicitly() {
    let dir = tempfile::tempdir().expect("temp dir");
    let bytes = Gguf::new().u32(4242).u64(0).u64(0).build();
    let path = write_file(dir.path(), "future.gguf", &bytes);
    let mut registry = registry();

    match registry.register(&path).expect_err("must be rejected") {
        RegistryError::NotAnArtifact { source, .. } => match source {
            GgufError::UnsupportedVersion { version } => assert_eq!(version, 4242),
            other => panic!("expected UnsupportedVersion, got {other}"),
        },
        other => panic!("expected NotAnArtifact, got {other}"),
    }
}

#[test]
fn a_declared_length_beyond_the_file_is_refused_safely() {
    // A parser that allocated before checking would try to reserve terabytes here.
    let dir = tempfile::tempdir().expect("temp dir");
    let bytes = Gguf::new().u32(3).u64(0).u64(1).u64(u64::MAX).build();
    let path = write_file(dir.path(), "huge.gguf", &bytes);
    let mut registry = registry();

    match registry.register(&path).expect_err("must be rejected") {
        RegistryError::NotAnArtifact { source, .. } => assert!(
            matches!(source, GgufError::Implausible { .. }),
            "expected Implausible, got {source}"
        ),
        other => panic!("expected NotAnArtifact, got {other}"),
    }
}

#[test]
fn an_absurd_entry_count_is_refused_before_allocating() {
    let dir = tempfile::tempdir().expect("temp dir");
    let bytes = Gguf::new().u32(3).u64(0).u64(u64::MAX).build();
    let path = write_file(dir.path(), "many.gguf", &bytes);
    let mut registry = registry();

    match registry.register(&path).expect_err("must be rejected") {
        RegistryError::NotAnArtifact { source, .. } => assert!(
            matches!(source, GgufError::Implausible { .. }),
            "expected Implausible, got {source}"
        ),
        other => panic!("expected NotAnArtifact, got {other}"),
    }
}

#[test]
fn a_missing_path_is_reported_as_missing() {
    let mut registry = registry();
    let err = registry
        .register("no-such-model-anywhere.gguf")
        .expect_err("must be rejected");
    assert!(
        matches!(err, RegistryError::NotFound { .. }),
        "expected NotFound, got {err}"
    );
}

#[test]
fn a_directory_is_rejected() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut registry = registry();
    let err = registry.register(dir.path()).expect_err("must be rejected");
    assert!(
        matches!(err, RegistryError::NotAFile { .. }),
        "expected NotAFile, got {err}"
    );
}

#[test]
fn registering_the_same_path_twice_is_idempotent() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &valid_container());
    let mut registry = registry();

    let first = registry.register(&path).expect("registers");
    let second = registry.register(&path).expect("registers again");

    assert!(first.is_new());
    assert!(!second.is_new(), "the second call must not create a record");
    assert_eq!(
        first.artifact().id,
        second.artifact().id,
        "the identifier must be stable across repeated registration"
    );
    assert_eq!(registry.list().expect("lists").len(), 1);
}

#[test]
fn path_aliases_resolve_to_one_artifact() {
    // Reaching the same file by a different route must not create a second record,
    // or the registry would hold two identifiers for one artifact.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &valid_container());
    let alias = dir.path().join(".").join("model.gguf");
    let mut registry = registry();

    let first = registry.register(&path).expect("registers");
    let second = registry.register(&alias).expect("registers via alias");

    assert_eq!(first.artifact().id, second.artifact().id);
    assert_eq!(registry.list().expect("lists").len(), 1);
}

#[test]
fn the_recorded_path_is_canonical() {
    let dir = tempfile::tempdir().expect("temp dir");
    write_file(dir.path(), "model.gguf", &valid_container());
    let alias = dir.path().join(".").join("model.gguf");
    let mut registry = registry();

    let artifact = registry
        .register(&alias)
        .expect("registers")
        .artifact()
        .clone();
    assert!(
        !artifact.path.to_string_lossy().contains("\\.\\")
            && !artifact.path.to_string_lossy().contains("/./"),
        "the recorded path should be canonical, got {}",
        artifact.path.display()
    );
}

#[test]
fn a_replaced_file_is_reported_as_changed_rather_than_adopted() {
    // Silently updating the record would erase the difference between the artifact
    // that was inspected and whatever now sits at that path.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &valid_container());
    let mut registry = registry();
    let id = registry
        .register(&path)
        .expect("registers")
        .artifact()
        .id
        .clone();

    assert_eq!(
        registry.check(&id).expect("checks"),
        ArtifactState::Unchanged
    );

    // Replace with a different container: same name, different contents and size.
    let replacement = Gguf::new()
        .u32(3)
        .u64(0)
        .u64(1)
        .text_entry("general.architecture", "something-else")
        .build();
    write_file(dir.path(), "model.gguf", &replacement);

    match registry.check(&id).expect("checks") {
        ArtifactState::Changed { reason } => assert!(reason.contains("size"), "{reason}"),
        other => panic!("expected Changed, got {other}"),
    }

    // The record still describes what was originally inspected.
    let recorded = registry.get(&id).expect("reads").expect("present");
    assert_eq!(recorded.metadata.architecture.as_deref(), Some("llama"));
}

#[test]
fn a_deleted_file_is_reported_as_missing() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &valid_container());
    let mut registry = registry();
    let id = registry
        .register(&path)
        .expect("registers")
        .artifact()
        .id
        .clone();

    std::fs::remove_file(&path).expect("removes");
    assert_eq!(registry.check(&id).expect("checks"), ArtifactState::Missing);
}

#[test]
fn the_registry_survives_being_reopened() {
    let dir = tempfile::tempdir().expect("temp dir");
    let model = write_file(dir.path(), "model.gguf", &valid_container());
    let database = dir.path().join("registry.sqlite");

    let id = {
        let mut registry = ArtifactRegistry::open(&database).expect("opens");
        registry
            .register(&model)
            .expect("registers")
            .artifact()
            .id
            .clone()
    };

    let reopened = ArtifactRegistry::open(&database).expect("reopens");
    let artifact = reopened
        .get(&id)
        .expect("reads")
        .expect("the artifact survived the restart");

    assert_eq!(
        artifact.id, id,
        "the identifier must be stable across a restart"
    );
    assert_eq!(artifact.metadata.architecture.as_deref(), Some("llama"));
    assert_eq!(artifact.metadata.context_length, Some(8192));
    assert_eq!(artifact.integrity, ArtifactIntegrity::Unverified);
}

#[test]
fn uninterpreted_metadata_is_kept_rather_than_discarded() {
    // A key the runtime does not reason about is still recorded, so learning a new
    // key later is not a schema change.
    let dir = tempfile::tempdir().expect("temp dir");
    let bytes = Gguf::new()
        .u32(3)
        .u64(0)
        .u64(2)
        .text_entry("general.architecture", "llama")
        .text_entry("some.vendor.specific.key", "vendor value")
        .build();
    let path = write_file(dir.path(), "model.gguf", &bytes);
    let mut registry = registry();
    let id = registry
        .register(&path)
        .expect("registers")
        .artifact()
        .id
        .clone();

    let metadata = registry.metadata(&id).expect("reads metadata");
    let found = metadata
        .iter()
        .find(|(key, _)| key == "some.vendor.specific.key")
        .expect("the uninterpreted key is retained");
    assert_eq!(found.1, "vendor value");
}

#[test]
fn verification_is_a_separate_step_from_registration() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &valid_container());
    let mut registry = registry();
    let id = registry
        .register(&path)
        .expect("registers")
        .artifact()
        .id
        .clone();

    assert_eq!(
        registry
            .get(&id)
            .expect("reads")
            .expect("present")
            .integrity,
        ArtifactIntegrity::Unverified
    );

    let integrity = registry.verify(&id).expect("verifies");
    match integrity {
        ArtifactIntegrity::Verified { digest, .. } => assert_eq!(digest.len(), 32),
        ArtifactIntegrity::Unverified => panic!("verification must produce a digest"),
    }

    assert!(
        registry
            .get(&id)
            .expect("reads")
            .expect("present")
            .integrity
            .is_verified(),
        "the verified state must be durable"
    );
}

#[test]
fn forgetting_an_artifact_leaves_the_file_alone() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = write_file(dir.path(), "model.gguf", &valid_container());
    let mut registry = registry();
    let id = registry
        .register(&path)
        .expect("registers")
        .artifact()
        .id
        .clone();

    assert!(registry.forget(&id).expect("forgets"));
    assert!(registry.get(&id).expect("reads").is_none());
    assert!(
        path.exists(),
        "registration records a file; forgetting must not delete it"
    );
}

/// Real containers present on this machine, if any.
fn real_containers() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    let roots = [
        PathBuf::from(&home).join(".lmstudio/models"),
        PathBuf::from(&home).join(".lmstudio/.internal/bundled-models"),
    ];
    for root in roots {
        collect_gguf(&root, &mut found, 0);
    }
    found
}

fn collect_gguf(dir: &Path, found: &mut Vec<PathBuf>, depth: usize) {
    if depth > 4 || found.len() >= 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf(&path, found, depth + 1);
        } else if path.extension().is_some_and(|ext| ext == "gguf") {
            found.push(path);
        }
    }
}

#[test]
fn real_containers_on_this_machine_are_understood() {
    let containers = real_containers();
    if containers.is_empty() {
        eprintln!("SKIPPED: no real GGUF containers found on this machine");
        return;
    }

    let mut registry = registry();
    for path in containers {
        let artifact = registry
            .register(&path)
            .unwrap_or_else(|err| panic!("{} did not register: {err}", path.display()))
            .artifact()
            .clone();

        assert!(
            artifact.metadata.architecture.is_some(),
            "{} reported no architecture",
            path.display()
        );
        assert!(artifact.size_bytes > 0);
        assert_eq!(artifact.integrity, ArtifactIntegrity::Unverified);
        eprintln!(
            "registered {}: architecture {:?}, {} tensors, context {:?}, chat template {}",
            path.file_name().unwrap_or_default().to_string_lossy(),
            artifact.metadata.architecture.as_deref().unwrap_or("?"),
            artifact.metadata.tensor_count,
            artifact.metadata.context_length,
            artifact.metadata.has_chat_template()
        );
    }
}

#[test]
fn inspecting_a_large_container_does_not_read_it_all() {
    // Registration must cost the same for a 5 GB container as for a small one. If
    // it ever starts reading tensor data, this becomes visibly slow.
    let containers = real_containers();
    let Some(largest) = containers
        .iter()
        .max_by_key(|path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
    else {
        eprintln!("SKIPPED: no real GGUF containers found on this machine");
        return;
    };

    let size = std::fs::metadata(largest).expect("stats").len();
    if size < 512 * 1024 * 1024 {
        eprintln!("SKIPPED: largest container is only {size} bytes");
        return;
    }

    let mut registry = registry();
    let began = std::time::Instant::now();
    registry.register(largest).expect("registers");
    let elapsed = began.elapsed();

    eprintln!("inspected {} MB in {elapsed:?}", size / (1024 * 1024));
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "inspecting {} MB took {elapsed:?}, which suggests the whole file is being read",
        size / (1024 * 1024)
    );
}

#[cfg(windows)]
fn backend_process_count() -> usize {
    let output = std::process::Command::new("tasklist")
        .args(["/FI", "IMAGENAME eq llama-server.exe", "/NH", "/FO", "CSV"])
        .output();
    output.map_or(0, |output| {
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| line.contains("llama-server"))
            .count()
    })
}

#[cfg(unix)]
fn backend_process_count() -> usize {
    let output = std::process::Command::new("pgrep")
        .args(["-c", "llama-server"])
        .output();
    output.map_or(0, |output| {
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .unwrap_or(0)
    })
}

#[test]
fn registration_never_launches_a_backend() {
    // Knowing what an artifact is must not depend on being able to execute it.
    // The registry crate has no dependency on any backend, so this cannot happen
    // by construction; the test guards the property against a future change that
    // would reintroduce it.
    let before = backend_process_count();

    let dir = tempfile::tempdir().expect("temp dir");
    let mut registry = registry();
    for container in real_containers().into_iter().take(2) {
        let _ = registry.register(&container);
    }
    let synthetic = write_file(dir.path(), "model.gguf", &valid_container());
    registry.register(&synthetic).expect("registers");

    let after = backend_process_count();
    assert_eq!(
        before,
        after,
        "registering an artifact started {} backend process(es)",
        after.saturating_sub(before)
    );
}
