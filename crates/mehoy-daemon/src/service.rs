//! Request routing for the daemon's local HTTP surface.
//!
//! The surface is deliberately tiny. ADR-0003 keeps the first protocol to runtime
//! metadata only, so that the implementation grows from the runtime's own model
//! rather than outward from a compatibility endpoint.
//!
//! Routing is a pure function of method and path so it can be tested without a
//! live connection.

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::{Method, Response, StatusCode, header};

use mehoy_protocol::{
    ErrorBody, Health, HealthStatus, PATH_HEALTH, PATH_RUNTIME, ProtocolVersion, RuntimeIdentity,
    RuntimeInfo,
};

/// Name this daemon reports as its own.
pub const RUNTIME_NAME: &str = "mehoyd";

/// Version this daemon reports, taken from the package version so the two cannot
/// drift apart.
pub const RUNTIME_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Serves one request, given its method and path.
#[must_use]
pub fn route(method: &Method, path: &str) -> Response<Full<Bytes>> {
    match path {
        PATH_HEALTH => match *method {
            Method::GET => json(
                StatusCode::OK,
                &Health {
                    status: HealthStatus::Ok,
                },
            ),
            _ => method_not_allowed(),
        },
        PATH_RUNTIME => match *method {
            Method::GET => json(StatusCode::OK, &runtime_info()),
            _ => method_not_allowed(),
        },
        _ => json(
            StatusCode::NOT_FOUND,
            &ErrorBody::new("not_found", format!("no route for {path}")),
        ),
    }
}

/// This daemon's identity and the protocol version it speaks.
#[must_use]
pub fn runtime_info() -> RuntimeInfo {
    RuntimeInfo {
        protocol: ProtocolVersion::CURRENT,
        runtime: RuntimeIdentity {
            name: RUNTIME_NAME.to_owned(),
            version: RUNTIME_VERSION.to_owned(),
        },
    }
}

fn method_not_allowed() -> Response<Full<Bytes>> {
    let mut response = json(
        StatusCode::METHOD_NOT_ALLOWED,
        &ErrorBody::new("method_not_allowed", "this route accepts GET only"),
    );
    response
        .headers_mut()
        .insert(header::ALLOW, header::HeaderValue::from_static("GET"));
    response
}

/// Builds a JSON response.
///
/// Serialization of these types cannot fail, but the fallback avoids a panic in
/// a long-running process if that ever stops being true.
fn json<T: serde::Serialize>(status: StatusCode, body: &T) -> Response<Full<Bytes>> {
    match serde_json::to_vec(body) {
        Ok(bytes) => build(status, bytes),
        Err(_) => build(
            StatusCode::INTERNAL_SERVER_ERROR,
            br#"{"error":{"code":"encoding_failed","message":"response could not be encoded"}}"#
                .to_vec(),
        ),
    }
}

fn build(status: StatusCode, bytes: Vec<u8>) -> Response<Full<Bytes>> {
    let mut response = Response::new(Full::new(Bytes::from(bytes)));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    async fn body_of(response: Response<Full<Bytes>>) -> serde_json::Value {
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body collects")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("body is json")
    }

    #[tokio::test]
    async fn health_reports_ok() {
        let response = route(&Method::GET, PATH_HEALTH);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        assert_eq!(body_of(response).await, serde_json::json!({"status": "ok"}));
    }

    #[tokio::test]
    async fn runtime_reports_protocol_and_identity() {
        let response = route(&Method::GET, PATH_RUNTIME);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            body_of(response).await,
            serde_json::json!({
                "protocol": { "major": 1, "minor": 0 },
                "runtime": { "name": "mehoyd", "version": RUNTIME_VERSION }
            })
        );
    }

    #[tokio::test]
    async fn unknown_path_is_not_found_with_a_coded_error() {
        let response = route(&Method::GET, "/nope");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = body_of(response).await;
        assert_eq!(body["error"]["code"], "not_found");
    }

    #[tokio::test]
    async fn non_get_on_a_known_route_is_rejected_and_advertises_get() {
        for method in [Method::POST, Method::PUT, Method::DELETE] {
            let response = route(&method, PATH_HEALTH);
            assert_eq!(
                response.status(),
                StatusCode::METHOD_NOT_ALLOWED,
                "method {method} should be rejected"
            );
            assert_eq!(response.headers().get(header::ALLOW).unwrap(), "GET");
            let body = body_of(response).await;
            assert_eq!(body["error"]["code"], "method_not_allowed");
        }
    }

    #[test]
    fn no_compatibility_endpoint_is_served() {
        // ADR-0003 excludes a vendor-compatible endpoint from the first slice.
        // This guards against it being added without a decision to add it.
        for path in ["/v1/chat/completions", "/v1/completions", "/v1/models"] {
            let response = route(&Method::POST, path);
            assert_eq!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{path} must not be served yet"
            );
        }
    }

    #[test]
    fn reported_version_matches_the_package() {
        assert_eq!(runtime_info().runtime.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(runtime_info().runtime.name, "mehoyd");
    }
}
