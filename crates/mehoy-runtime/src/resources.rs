//! What a resident model is accounted as occupying, and what a runtime will hold.
//!
//! # Declared, not measured
//!
//! Every number here is declared by the artifact, not measured on the machine.
//! The runtime knows how large a container is because it registered the file; it
//! does not know how much memory this host has, how much is free, or how much of
//! the model the backend will place on an accelerator. Discovering any of that is
//! separate work, and until it exists an accounting layer that implied it would be
//! worse than none at all.
//!
//! So a footprint carries its own basis, in the same spirit as capability
//! evidence: the number is close to useless without knowing where it came from,
//! and a reader who cannot tell a declared figure from a measured one will treat
//! both as fact.
//!
//! # What the accounting is therefore good for
//!
//! Stating a policy and enforcing it consistently. An operator who knows their
//! machine can say how many models and how many bytes of weights this runtime may
//! hold, and every load is then refused or admitted against that one statement
//! rather than against whatever each call site happened to assume. What it is not
//! good for is deciding whether a model will actually fit in the hardware, which
//! requires knowing the hardware.

use std::fmt;

/// How many models a runtime holds when nothing says otherwise.
///
/// One. Lifting it is now a configuration choice rather than a code change, but
/// the default stays at one because this runtime cannot yet tell whether a second
/// model would fit: see the module documentation on declared numbers.
pub const DEFAULT_RESIDENT_CAPACITY: usize = 1;

/// How a footprint figure was arrived at.
///
/// Recorded alongside the figure so nothing has to guess how much weight it
/// carries. One variant today, because the runtime has exactly one source of
/// truth: what it read off the container when it registered it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FootprintBasis {
    /// The size of the container on disk, as recorded at registration.
    ///
    /// A floor on what must be resident somewhere, not a total. It excludes the
    /// key/value cache, the backend's compute buffers and the backend process
    /// itself, and it says nothing about whether the bytes land in host memory or
    /// on an accelerator.
    ContainerSize,
}

impl fmt::Display for FootprintBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContainerSize => f.write_str("the registered container size"),
        }
    }
}

/// What one model is accounted as occupying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceFootprint {
    /// Bytes of model weights.
    pub weight_bytes: u64,
    /// Where `weight_bytes` came from.
    pub basis: FootprintBasis,
}

impl ResourceFootprint {
    /// Accounts a model at the size of the container it loads from.
    #[must_use]
    pub const fn from_container_size(size_bytes: u64) -> Self {
        Self {
            weight_bytes: size_bytes,
            basis: FootprintBasis::ContainerSize,
        }
    }
}

impl fmt::Display for ResourceFootprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} byte(s) of weights, from {}",
            self.weight_bytes, self.basis
        )
    }
}

/// What a runtime will hold at once.
///
/// Both limits are stated by whoever builds the runtime. Neither is derived from
/// the machine, so neither is a promise that what fits here fits in hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentLimits {
    /// How many models may be resident, counting those still being brought up.
    pub max_instances: usize,
    /// How many bytes of weights may be resident, when that is bounded at all.
    ///
    /// `None` means unbounded, which is the default. An unbounded byte budget is
    /// the honest default: picking a number for a machine nothing has measured
    /// would be inventing a limit, and a limit nobody chose is worse than one that
    /// is openly absent.
    pub max_weight_bytes: Option<u64>,
}

impl Default for ResidentLimits {
    fn default() -> Self {
        Self {
            max_instances: DEFAULT_RESIDENT_CAPACITY,
            max_weight_bytes: None,
        }
    }
}

impl ResidentLimits {
    /// Limits that hold at most `max_instances` models and do not bound bytes.
    #[must_use]
    pub const fn instances(max_instances: usize) -> Self {
        Self {
            max_instances,
            max_weight_bytes: None,
        }
    }

    /// The same, additionally bounding the weights held.
    #[must_use]
    pub const fn with_weight_budget(mut self, max_weight_bytes: u64) -> Self {
        self.max_weight_bytes = Some(max_weight_bytes);
        self
    }

    /// Whether one more model of this footprint fits.
    ///
    /// The instance count is checked first, so a runtime over both limits reports
    /// the count deterministically rather than depending on field order.
    ///
    /// # Errors
    ///
    /// Returns the limit that would be exceeded.
    pub(crate) fn admits(
        self,
        committed: ResidentUsage,
        footprint: ResourceFootprint,
    ) -> Result<(), Refusal> {
        if committed.instances >= self.max_instances {
            return Err(Refusal::Instances {
                capacity: self.max_instances,
            });
        }

        if let Some(limit) = self.max_weight_bytes {
            let requested = footprint.weight_bytes;
            // Checked rather than wrapping, and checked rather than saturating. A
            // total that saturated to the maximum would compare as fitting a
            // maximum-sized budget, which is exactly the case where it does not,
            // and a wrapped total would make an enormous request look tiny. A sum
            // that cannot be represented cannot fit any budget.
            let fits = committed
                .weight_bytes
                .checked_add(requested)
                .is_some_and(|total| total <= limit);
            if !fits {
                return Err(Refusal::WeightBytes {
                    requested,
                    committed: committed.weight_bytes,
                    limit,
                });
            }
        }

        Ok(())
    }
}

impl fmt::Display for ResidentLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at most {} model(s)", self.max_instances)?;
        match self.max_weight_bytes {
            Some(limit) => write!(f, " and {limit} byte(s) of weights"),
            None => f.write_str(", with no byte budget"),
        }
    }
}

/// What is committed against a runtime's limits right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResidentUsage {
    /// Models this runtime is holding, including those still being brought up.
    pub instances: usize,
    /// How many of `instances` have not finished loading.
    ///
    /// Reported separately because it answers the question a refusal otherwise
    /// provokes: why a load was turned away when nothing appears to be resident.
    pub loading: usize,
    /// Bytes of weights committed, including loads in flight.
    pub weight_bytes: u64,
}

impl fmt::Display for ResidentUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} model(s), {} still loading, {} byte(s) of weights",
            self.instances, self.loading, self.weight_bytes
        )
    }
}

/// Which limit a load would exceed.
///
/// Internal to the crate. The runtime translates it into its own error
/// vocabulary, so a caller matching on a refusal does not have to know that a
/// separate policy type exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refusal {
    Instances {
        capacity: usize,
    },
    WeightBytes {
        requested: u64,
        committed: u64,
        limit: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn footprint(bytes: u64) -> ResourceFootprint {
        ResourceFootprint::from_container_size(bytes)
    }

    fn holding(instances: usize, weight_bytes: u64) -> ResidentUsage {
        ResidentUsage {
            instances,
            loading: 0,
            weight_bytes,
        }
    }

    #[test]
    fn a_footprint_records_where_its_number_came_from() {
        // The whole point of the basis field. A bare byte count would be read as a
        // measurement of the machine, which it is not.
        let declared = footprint(5 * GIB);
        assert_eq!(declared.weight_bytes, 5 * GIB);
        assert_eq!(declared.basis, FootprintBasis::ContainerSize);
    }

    #[test]
    fn the_default_holds_one_model_and_does_not_bound_bytes() {
        let limits = ResidentLimits::default();
        assert_eq!(limits.max_instances, DEFAULT_RESIDENT_CAPACITY);
        assert_eq!(limits.max_instances, 1);
        assert_eq!(
            limits.max_weight_bytes, None,
            "a byte budget was invented for a machine nothing has measured"
        );
    }

    #[test]
    fn an_empty_runtime_admits_a_model() {
        assert!(
            ResidentLimits::default()
                .admits(ResidentUsage::default(), footprint(5 * GIB))
                .is_ok()
        );
    }

    #[test]
    fn a_full_runtime_refuses_by_count() {
        assert_eq!(
            ResidentLimits::default().admits(holding(1, 5 * GIB), footprint(GIB)),
            Err(Refusal::Instances { capacity: 1 })
        );
    }

    #[test]
    fn a_runtime_that_holds_nothing_admits_nothing() {
        // A legitimate configuration rather than a degenerate one: it is how an
        // operator says this process must not bring models up at all.
        assert_eq!(
            ResidentLimits::instances(0).admits(ResidentUsage::default(), footprint(1)),
            Err(Refusal::Instances { capacity: 0 })
        );
    }

    #[test]
    fn an_unbounded_byte_budget_admits_any_size() {
        assert!(
            ResidentLimits::instances(2)
                .admits(holding(1, u64::MAX - 1), footprint(u64::MAX))
                .is_ok(),
            "an absent budget was treated as a budget of some size"
        );
    }

    #[test]
    fn a_model_that_does_not_fit_what_is_left_is_refused() {
        let limits = ResidentLimits::instances(4).with_weight_budget(8 * GIB);
        assert_eq!(
            limits.admits(holding(1, 6 * GIB), footprint(3 * GIB)),
            Err(Refusal::WeightBytes {
                requested: 3 * GIB,
                committed: 6 * GIB,
                limit: 8 * GIB,
            })
        );
    }

    #[test]
    fn a_model_filling_the_budget_exactly_is_admitted() {
        // The boundary, stated: a budget is what may be held, not what must be
        // stayed under.
        let limits = ResidentLimits::instances(4).with_weight_budget(8 * GIB);
        assert!(
            limits
                .admits(holding(1, 6 * GIB), footprint(2 * GIB))
                .is_ok()
        );
        assert!(
            limits
                .admits(holding(1, 6 * GIB), footprint(2 * GIB + 1))
                .is_err()
        );
    }

    #[test]
    fn a_model_too_large_for_the_whole_budget_is_refused_from_empty() {
        // Distinguishable from the previous case by its own fields: a request
        // exceeding the limit with nothing committed will never fit here, however
        // much is unloaded first.
        let limits = ResidentLimits::instances(4).with_weight_budget(4 * GIB);
        let refusal = limits
            .admits(ResidentUsage::default(), footprint(5 * GIB))
            .expect_err("a model larger than the budget must be refused");
        assert_eq!(
            refusal,
            Refusal::WeightBytes {
                requested: 5 * GIB,
                committed: 0,
                limit: 4 * GIB,
            }
        );
    }

    #[test]
    fn a_count_refusal_is_reported_ahead_of_a_byte_refusal() {
        // Both limits are exceeded. Which one is reported is fixed here rather
        // than left to the order the fields happen to be checked in.
        let limits = ResidentLimits::instances(1).with_weight_budget(GIB);
        assert_eq!(
            limits.admits(holding(1, GIB), footprint(5 * GIB)),
            Err(Refusal::Instances { capacity: 1 })
        );
    }

    #[test]
    fn a_total_that_cannot_be_represented_is_refused_rather_than_wrapped() {
        // The budget is the largest one expressible, so saturating the sum would
        // make the total compare as fitting precisely when it overflowed.
        let limits = ResidentLimits::instances(4).with_weight_budget(u64::MAX);
        assert_eq!(
            limits.admits(holding(1, u64::MAX - 10), footprint(100)),
            Err(Refusal::WeightBytes {
                requested: 100,
                committed: u64::MAX - 10,
                limit: u64::MAX,
            })
        );
    }

    #[test]
    fn loading_models_are_part_of_what_is_held() {
        // Not a property of this function so much as of the vocabulary it uses:
        // `instances` counts loads in flight, so a reservation refuses the next
        // caller exactly as a finished load would.
        let committed = ResidentUsage {
            instances: 1,
            loading: 1,
            weight_bytes: 5 * GIB,
        };
        assert_eq!(
            ResidentLimits::default().admits(committed, footprint(GIB)),
            Err(Refusal::Instances { capacity: 1 })
        );
    }
}
