//! A stand-in for the engine, enough for the runtime to supervise and talk to.
//!
//! The runtime's own behaviour (admission, the resource ledger, instance
//! ownership, unload draining) could only be tested on a machine holding a real
//! engine and a multi-gigabyte container, because `LlamaCppBackend` takes an
//! executable path and there is no seam for anything else. That made the tests
//! for this runtime's logic skip almost everywhere, including on the Linux run,
//! where what they actually proved was compilation.
//!
//! This program closes that gap from the outside rather than by adding a seam:
//! it is an executable, so pointing `MEHOY_LLAMA_SERVER` at it needs no change to
//! the production path at all.
//!
//! # What it deliberately does not do
//!
//! Inference. It returns responses with the right shape and no meaning. A
//! capability is verified by performing a task (ADR-0005, and the evidence ladder
//! in `mehoy-runtime::capability`), so a stand-in that let a capability reach
//! `Verified` would be a lie in the one place this project is most careful about.
//! Tests using it must not promote a capability.
//!
//! # Behaviour
//!
//! The runtime builds the command line, so a test cannot pass a flag. It names
//! the container, though, and the container is what `--model` carries, so the
//! behaviour travels in the file name: everything after a `__` in the stem.
//!
//! ```text
//! fixture.gguf                          bind, answer, stay up
//! fixture__slow-ready-400.gguf          wait 400 ms before binding
//! fixture__never-ready.gguf             stay alive and never bind
//! fixture__refuse-credential.gguf       bind, but reject every credential
//! fixture__fail-load.gguf               report a load failure and exit non-zero
//! fixture__exit-after-serving-50.gguf   answer one request, then abandon the instance
//! ```
//!
//! An environment variable would have been the obvious choice and is the wrong
//! one: it is process-wide, so two tests running concurrently in one binary would
//! take each other's behaviour. This way it belongs to the instance.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Reported by `--version`, in the shape the identity parser expects.
///
/// Deliberately says what it is. An identity recorded against a verification
/// should never leave a reader guessing whether a real engine was involved.
const BANNER: &str = "mehoy stand-in backend, not an inference engine\nversion: 0 (stand-in)";

/// Separates a fixture's name from the behaviour it asks for.
const BEHAVIOUR_MARKER: &str = "__";

/// How wide a stand-in embedding is. Small, fixed, and consistent across inputs,
/// which is all the runtime's structural validation asks of it.
const EMBEDDING_WIDTH: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Behaviour {
    Serve,
    SlowReady { after: Duration },
    NeverReady,
    RefuseCredential,
    FailLoad,
    ExitAfterServing { after: Duration },
}

impl Behaviour {
    /// Reads the behaviour out of the container's file name.
    fn from_model(model: Option<&PathBuf>) -> Result<Self, String> {
        let Some(stem) = model
            .and_then(|path| path.file_stem())
            .map(|stem| stem.to_string_lossy().into_owned())
        else {
            return Ok(Self::Serve);
        };
        let Some((_, request)) = stem.rsplit_once(BEHAVIOUR_MARKER) else {
            return Ok(Self::Serve);
        };

        // A trailing number is the delay, so `slow-ready-400` is four hundred
        // milliseconds. Split from the right, because the names contain hyphens.
        let numbered = |name: &str| -> Option<Duration> {
            request
                .strip_prefix(name)?
                .strip_prefix('-')?
                .parse::<u64>()
                .ok()
                .map(Duration::from_millis)
        };

        match request {
            "serve" => Ok(Self::Serve),
            "never-ready" => Ok(Self::NeverReady),
            "refuse-credential" => Ok(Self::RefuseCredential),
            "fail-load" => Ok(Self::FailLoad),
            _ => {
                if let Some(after) = numbered("slow-ready") {
                    Ok(Self::SlowReady { after })
                } else if let Some(after) = numbered("exit-after-serving") {
                    Ok(Self::ExitAfterServing { after })
                } else {
                    Err(format!(
                        "unrecognised behaviour {request:?} in the container name"
                    ))
                }
            }
        }
    }
}

/// What the runtime asked for on the command line.
#[derive(Debug, Default)]
struct Invocation {
    model: Option<PathBuf>,
    host: Option<String>,
    port: Option<u16>,
    api_key_file: Option<PathBuf>,
    embedding: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Invocation, String> {
    let mut parsed = Invocation::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || -> Result<String, String> {
            args.next().ok_or_else(|| format!("{arg} requires a value"))
        };
        match arg.as_str() {
            "--model" => parsed.model = Some(PathBuf::from(value()?)),
            "--host" => parsed.host = Some(value()?),
            "--port" => {
                let raw = value()?;
                parsed.port = Some(
                    raw.parse::<u16>()
                        .map_err(|_| format!("--port expects a port number, got {raw}"))?,
                );
            }
            "--api-key-file" => parsed.api_key_file = Some(PathBuf::from(value()?)),
            "--embedding" => parsed.embedding = true,
            "--no-webui" => {}
            // Accepted and ignored. They change how a real engine allocates, and
            // this one allocates nothing.
            "--ctx-size" | "--n-gpu-layers" => {
                let _ = value()?;
            }
            other => return Err(format!("unrecognised argument: {other}")),
        }
    }
    Ok(parsed)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--version") {
        // The engine writes its banner to standard error, and the identity probe
        // reads that stream, so this has to match.
        eprintln!("{BANNER}");
        return ExitCode::SUCCESS;
    }

    let invocation = match parse_args(args) {
        Ok(invocation) => invocation,
        Err(message) => {
            eprintln!("stand-in: {message}");
            return ExitCode::from(2);
        }
    };

    let behaviour = match Behaviour::from_model(invocation.model.as_ref()) {
        Ok(behaviour) => behaviour,
        Err(message) => {
            eprintln!("stand-in: {message}");
            return ExitCode::from(2);
        }
    };

    if behaviour == Behaviour::FailLoad {
        eprintln!("stand-in: refusing to load the model, as requested");
        return ExitCode::from(1);
    }

    // A real engine fails here, so this one does too: a container that is not
    // there is the most ordinary load failure there is.
    if let Some(model) = &invocation.model {
        if !model.is_file() {
            eprintln!(
                "stand-in: cannot load model {}: no such file",
                model.display()
            );
            return ExitCode::from(1);
        }
    }

    let credential = match &invocation.api_key_file {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(contents) => contents.trim().to_owned(),
            Err(err) => {
                eprintln!("stand-in: cannot read {}: {err}", path.display());
                return ExitCode::from(1);
            }
        },
        None => String::new(),
    };

    if behaviour == Behaviour::NeverReady {
        eprintln!("stand-in: staying up without binding, as requested");
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    if let Behaviour::SlowReady { after } = behaviour {
        eprintln!("stand-in: delaying readiness by {} ms", after.as_millis());
        tokio::time::sleep(after).await;
    }

    let Some(port) = invocation.port else {
        eprintln!("stand-in: --port is required");
        return ExitCode::from(2);
    };
    let host = invocation.host.as_deref().unwrap_or("127.0.0.1");
    let Ok(ip) = host.parse::<IpAddr>() else {
        eprintln!("stand-in: --host expects an address, got {host}");
        return ExitCode::from(2);
    };

    let listener = match TcpListener::bind(SocketAddr::new(ip, port)).await {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("stand-in: cannot bind {host}:{port}: {err}");
            return ExitCode::from(1);
        }
    };
    // Printed for the captured output a failure report would include, and because
    // it is the only way to tell from a log which one of these was running.
    eprintln!(
        "stand-in: listening on {host}:{port}, embedding={}",
        invocation.embedding
    );

    let served = Arc::new(AtomicBool::new(false));
    let state = State {
        credential,
        refuse_credential: behaviour == Behaviour::RefuseCredential,
        served: Some(Arc::clone(&served)),
    };

    if let Behaviour::ExitAfterServing { after } = behaviour {
        let serving = tokio::spawn(accept_loop(listener, state));
        // Triggered by an inference request rather than by a delay, because the
        // delay has to land after the load and nothing here can see when that
        // finished. Timing it from the bind, or from the credential probe, races
        // the rest of startup, and the supervisor then reports a worker that
        // exited during startup instead of an instance whose backend abandoned it.
        while !served.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        tokio::time::sleep(after).await;
        serving.abort();
        eprintln!("stand-in: exiting after serving, as requested");
        return ExitCode::SUCCESS;
    }

    accept_loop(listener, state).await;
    ExitCode::SUCCESS
}

#[derive(Clone)]
struct State {
    credential: String,
    refuse_credential: bool,
    /// Set once an inference request has been answered, which is strictly after
    /// the runtime finished loading and took the instance into service.
    served: Option<Arc<AtomicBool>>,
}

impl State {
    fn mark_served(&self) {
        if let Some(flag) = &self.served {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

async fn accept_loop(listener: TcpListener, state: State) {
    loop {
        let Ok((stream, _peer)) = listener.accept().await else {
            continue;
        };
        let state = state.clone();
        tokio::spawn(async move {
            let _ = serve_connection(stream, state).await;
        });
    }
}

/// One request, one response, then close.
///
/// No keep-alive, which is safe because every adapter path opens its own
/// connection: generation documents that it never pools, since closing the
/// connection is how that backend is asked to stop.
async fn serve_connection(mut stream: TcpStream, state: State) -> std::io::Result<()> {
    let Some(request) = read_request(&mut stream).await? else {
        return Ok(());
    };

    let response = route(&request, &state);
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await
}

struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

impl Request {
    fn bearer(&self) -> Option<&str> {
        self.headers
            .get("authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
    }

    fn wants_stream(&self) -> bool {
        // Matched on the text rather than parsed, because the stand-in has no
        // business having an opinion about the rest of the body.
        self.body.contains("\"stream\":true") || self.body.contains("\"stream\": true")
    }
}

async fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut raw = Vec::new();
    let mut byte = [0u8; 1];
    // Bounded, so a client that never sends a blank line cannot grow this without
    // limit. The head of a request from this runtime is a few hundred bytes.
    while raw.len() < 16 * 1024 {
        if stream.read_exact(&mut byte).await.is_err() {
            return Ok(None);
        }
        raw.push(byte[0]);
        if raw.ends_with(b"\r\n\r\n") {
            break;
        }
    }

    let head = String::from_utf8_lossy(&raw).into_owned();
    let mut lines = head.lines();
    let Some(start) = lines.next() else {
        return Ok(None);
    };
    let mut parts = start.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }

    let length: usize = headers
        .get("content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; length.min(8 * 1024 * 1024)];
    if !body.is_empty() && stream.read_exact(&mut body).await.is_err() {
        return Ok(None);
    }

    Ok(Some(Request {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    }))
}

fn route(request: &Request, state: &State) -> String {
    match (request.method.as_str(), request.path.as_str()) {
        // Unauthenticated on the real engine, and measured to be so. The runtime
        // relies on that difference from /props to tell a wrong credential from a
        // backend that is merely still starting.
        ("GET", "/health") => json(200, "OK", r#"{"status":"ok"}"#),
        ("GET", "/props") => {
            if state.refuse_credential || request.bearer() != Some(state.credential.as_str()) {
                unauthorised()
            } else {
                json(
                    200,
                    "OK",
                    r#"{"default_generation_settings":{"n_ctx":4096},"model_path":"stand-in"}"#,
                )
            }
        }
        ("POST", "/v1/completions") => {
            if request.bearer() != Some(state.credential.as_str()) {
                return unauthorised();
            }
            state.mark_served();
            if request.wants_stream() {
                streamed_completion()
            } else {
                json(200, "OK", &completion_body())
            }
        }
        ("POST", "/v1/embeddings") => {
            if request.bearer() != Some(state.credential.as_str()) {
                return unauthorised();
            }
            state.mark_served();
            json(200, "OK", &embedding_body(&request.body))
        }
        _ => json(
            404,
            "Not Found",
            r#"{"error":{"message":"no such route on the stand-in","type":"not_found"}}"#,
        ),
    }
}

fn json(status: u16, reason: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn unauthorised() -> String {
    json(
        401,
        "Unauthorized",
        r#"{"error":{"message":"the stand-in refused the credential","type":"authentication_error"}}"#,
    )
}

fn completion_body() -> String {
    format!(
        r#"{{"choices":[{{"text":"{STAND_IN_TEXT}","finish_reason":"stop"}}],"usage":{{"prompt_tokens":1,"completion_tokens":{}}}}}"#,
        STAND_IN_TEXT.split_whitespace().count()
    )
}

/// What a completion returns. Not an answer to anything.
const STAND_IN_TEXT: &str = " a stand-in response, which means nothing";

/// Server-sent events in the shape the decoder expects: a delta per event, then
/// a final event carrying the finish reason, then the done marker.
fn streamed_completion() -> String {
    let mut out = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
    );
    for word in STAND_IN_TEXT.split_inclusive(' ') {
        out.push_str(&format!(
            "data: {{\"choices\":[{{\"text\":\"{word}\"}}]}}\n\n"
        ));
    }
    out.push_str(
        "data: {\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":7}}\n\n",
    );
    out.push_str("data: [DONE]\n\n");
    out
}

/// One vector per input, derived from the input so it is stable, with a fixed
/// width so the runtime's dimension check has something consistent to agree with.
fn embedding_body(body: &str) -> String {
    let inputs = count_inputs(body);
    let mut data = String::new();
    for index in 0..inputs {
        if index > 0 {
            data.push(',');
        }
        let seed = (index as f32).mul_add(0.125, 0.25);
        let vector: Vec<String> = (0..EMBEDDING_WIDTH)
            .map(|axis| format!("{:.6}", seed + (axis as f32) * 0.015_625))
            .collect();
        data.push_str(&format!(
            r#"{{"index":{index},"embedding":[{}]}}"#,
            vector.join(",")
        ));
    }
    format!(r#"{{"object":"list","data":[{data}]}}"#)
}

/// How many strings the request's `input` array holds.
///
/// Counted rather than parsed, to keep this program free of a JSON dependency it
/// would otherwise need only here. The request is built by this workspace, so its
/// shape is known: `{"input":["...","..."]}`.
fn count_inputs(body: &str) -> usize {
    let Some(start) = body.find("\"input\":[") else {
        return 1;
    };
    let rest = &body[start + "\"input\":[".len()..];
    let Some(end) = rest.find(']') else {
        return 1;
    };
    let array = &rest[..end];
    if array.trim().is_empty() {
        return 0;
    }
    // Quote pairs rather than commas, because a comma may appear inside a string.
    array.matches('"').count() / 2
}
