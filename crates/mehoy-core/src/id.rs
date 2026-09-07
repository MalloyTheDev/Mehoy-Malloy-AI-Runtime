//! Opaque runtime identifiers.
//!
//! These are deliberately not exposed on the wire yet. Keeping them opaque now
//! means the representation can change without breaking a published contract.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

macro_rules! opaque_id {
    ($(#[$meta:meta])* $name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u64);

        impl $name {
            /// Wraps a raw value. Prefer [`IdAllocator`] over calling this directly.
            #[must_use]
            pub const fn from_raw(raw: u64) -> Self {
                Self(raw)
            }

            /// Returns the underlying value.
            #[must_use]
            pub const fn as_raw(self) -> u64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}-{}", $prefix, self.0)
            }
        }
    };
}

opaque_id!(
    /// Identifies a single inference request for its whole lifetime.
    RequestId,
    "req"
);

opaque_id!(
    /// Identifies a supervised backend worker process.
    WorkerId,
    "wrk"
);

opaque_id!(
    /// Identifies a model known to the runtime.
    ModelId,
    "mdl"
);

/// Hands out monotonically increasing identifiers.
///
/// Identifiers are unique within one daemon process. They are not stable across
/// restarts and must not be persisted or treated as durable references.
#[derive(Debug, Default)]
pub struct IdAllocator {
    next: AtomicU64,
}

impl IdAllocator {
    /// Creates an allocator whose first identifier is `1`.
    ///
    /// Zero is skipped so that a defaulted or zeroed identifier is never mistaken
    /// for one that was actually allocated.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next: AtomicU64::new(1),
        }
    }

    fn next_raw(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    /// Allocates the next request identifier.
    pub fn request(&self) -> RequestId {
        RequestId::from_raw(self.next_raw())
    }

    /// Allocates the next worker identifier.
    pub fn worker(&self) -> WorkerId {
        WorkerId::from_raw(self.next_raw())
    }

    /// Allocates the next model identifier.
    pub fn model(&self) -> ModelId {
        ModelId::from_raw(self.next_raw())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_starts_at_one() {
        let alloc = IdAllocator::new();
        assert_eq!(alloc.request().as_raw(), 1);
    }

    #[test]
    fn allocator_does_not_repeat_across_kinds() {
        let alloc = IdAllocator::new();
        let a = alloc.request().as_raw();
        let b = alloc.worker().as_raw();
        let c = alloc.model().as_raw();
        assert_eq!([a, b, c], [1, 2, 3]);
    }

    #[test]
    fn display_is_prefixed_by_kind() {
        assert_eq!(RequestId::from_raw(7).to_string(), "req-7");
        assert_eq!(WorkerId::from_raw(7).to_string(), "wrk-7");
        assert_eq!(ModelId::from_raw(7).to_string(), "mdl-7");
    }

    #[test]
    fn ids_of_different_kinds_are_distinct_types() {
        // This test documents intent: the following must not compile.
        //   let _: RequestId = WorkerId::from_raw(1);
        let request = RequestId::from_raw(1);
        let worker = WorkerId::from_raw(1);
        assert_eq!(request.as_raw(), worker.as_raw());
        assert_ne!(request.to_string(), worker.to_string());
    }
}
