//! `amdid.f`'s table of the five best candidates and its parsimony pass.

use crate::ArimaModel;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RankedCandidate {
    pub(super) model: ArimaModel,
    pub(super) bic2: f64,
}

/// The five candidates with the smallest BIC2, best first.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct BestFive {
    slots: [Option<RankedCandidate>; 5],
}

impl BestFive {
    pub(super) fn slots(&self) -> &[Option<RankedCandidate>; 5] {
        &self.slots
    }

    pub(super) fn clear(&mut self) {
        self.slots.fill(None);
    }

    pub(super) fn contains(&self, model: ArimaModel) -> bool {
        self.slots
            .iter()
            .take_while(|slot| slot.is_some())
            .any(|slot| slot.as_ref().is_some_and(|entry| entry.model == model))
    }

    /// Inserts ahead of the first entry with a larger BIC2, so a tie stays
    /// behind the entries already ranked, and drops the sixth.
    pub(super) fn insert(&mut self, candidate: RankedCandidate) {
        if let Some(index) = self.slots.iter().position(|slot| {
            slot.as_ref()
                .is_none_or(|existing| existing.bic2 > candidate.bic2)
        }) {
            for move_from in (index..4).rev() {
                self.slots[move_from + 1] = self.slots[move_from].take();
            }
            self.slots[index] = Some(candidate);
        }
    }

    pub(super) fn best(&self) -> Option<&RankedCandidate> {
        self.slots[0].as_ref()
    }

    /// The model and BIC2 at `index`. Build 62 initializes the table to zero
    /// orders and a BIC2 of -999, which the parsimony pass reads for an empty
    /// slot.
    fn entry(&self, index: usize, zero_orders: ArimaModel) -> (ArimaModel, f64) {
        self.slots[index]
            .as_ref()
            .map_or((zero_orders, -999.0), |candidate| {
                (candidate.model, candidate.bic2)
            })
    }

    /// The parsimony pass: the best model, unless a lower-ranked one with
    /// fewer or cheaper terms scores within the thresholds. `zero_orders` is
    /// the model with no ARMA terms. `None` for an empty table.
    pub(super) fn select(&self, zero_orders: ArimaModel) -> Option<ArimaModel> {
        let best = self.best()?;
        let (mut chosen, best_score) = (best.model, best.bic2);
        let (_, fifth_score) = self.entry(4, zero_orders);
        let spread = (best_score - fifth_score).abs();
        let scale = if spread < 0.003 {
            0.0625
        } else if spread < 0.03 {
            0.25
        } else {
            1.0
        };
        let mut vc2 = 0.0025 * scale;
        let mut vc22 = 0.0075 * scale;
        let mut selected_index = 0;
        for index in 1..5 {
            let (alternative, score) = self.entry(index, zero_orders);
            let regular = chosen.regular();
            let seasonal = chosen.seasonal();
            let alt_regular = alternative.regular();
            let alt_seasonal = alternative.seasonal();
            let regular_terms = regular.p() + regular.q();
            let seasonal_terms = seasonal.p() + seasonal.q();
            let alt_regular_terms = alt_regular.p() + alt_regular.q();
            let alt_seasonal_terms = alt_seasonal.p() + alt_seasonal.q();
            let distance = (score - self.entry(selected_index, zero_orders).1).abs();
            let adds_first_regular_term = index == 1
                && regular_terms == 0
                && alt_regular_terms == 1
                && alt_seasonal_terms == seasonal_terms
                && distance < vc2;
            let reduces_regular = alt_regular_terms < regular_terms
                && alt_regular_terms > 0
                && alt_seasonal_terms == seasonal_terms
                && distance < vc2;
            let reduces_seasonal = alt_seasonal_terms < seasonal_terms
                && alt_seasonal_terms > 0
                && alt_regular_terms == regular_terms
                && distance < vc2;
            let reduces_regular_ar = alt_regular.p() < regular.p()
                && alt_regular.q() == regular.q()
                && alt_regular_terms > 0
                && alt_seasonal_terms == seasonal_terms
                && distance < vc22;
            if adds_first_regular_term || reduces_regular || reduces_seasonal || reduces_regular_ar
            {
                // Each switch tightens both thresholds by its distance from
                // the best score.
                let reduction = (best_score - score).abs();
                vc2 -= reduction;
                vc22 -= reduction;
                selected_index = index;
                chosen = alternative;
            }
        }
        let terms = chosen.regular().p()
            + chosen.regular().q()
            + chosen.seasonal().p()
            + chosen.seasonal().q();
        // A model without ARMA terms gives way to the next rank.
        if terms == 0 && selected_index < 4 {
            chosen = self.entry(selected_index + 1, zero_orders).0;
        }
        Some(chosen)
    }
}
