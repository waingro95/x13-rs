use serde::{Deserialize, Serialize};

use super::invalid_table;
use crate::OracleError;

/// The `(p, d, q)` orders of the model Build 62 reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SelectedModel {
    pub(crate) regular: [u8; 3],
    pub(crate) seasonal: [u8; 3],
}

/// Parses the model description Build 62's `mkmdsn.f` writes, such as
/// `(0 1 1)(0 1 1)`.
pub(crate) fn parse_selected_model(text: &str) -> Result<SelectedModel, OracleError> {
    let mut parser = ModelDescriptionParser::new(text);
    let regular = parser.parse_factor()?;
    parser.skip_whitespace();
    let seasonal = if parser.is_at_end() {
        [0; 3]
    } else {
        parser.parse_factor()?
    };
    parser.skip_whitespace();
    if !parser.is_at_end() {
        return Err(invalid_table(
            "automatic-model description has trailing text".to_owned(),
        ));
    }
    Ok(SelectedModel { regular, seasonal })
}

struct ModelDescriptionParser<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> ModelDescriptionParser<'a> {
    const fn new(text: &'a str) -> Self {
        Self { text, position: 0 }
    }

    fn parse_factor(&mut self) -> Result<[u8; 3], OracleError> {
        self.skip_whitespace();
        self.expect_byte(b'(')?;
        let mut orders = [0; 3];
        for (index, order) in orders.iter_mut().enumerate() {
            self.skip_whitespace();
            *order = self.parse_order()?;
            if index < 2 && !self.next_is_whitespace() {
                return Err(invalid_table(
                    "automatic-model orders must be whitespace-separated".to_owned(),
                ));
            }
        }
        self.skip_whitespace();
        self.expect_byte(b')')?;
        Ok(orders)
    }

    fn parse_order(&mut self) -> Result<u8, OracleError> {
        let start = self.position;
        while self
            .text
            .as_bytes()
            .get(self.position)
            .is_some_and(u8::is_ascii_digit)
        {
            self.position += 1;
        }
        if start == self.position {
            return Err(invalid_table(
                "automatic-model order is not an ASCII unsigned integer".to_owned(),
            ));
        }
        self.text[start..self.position]
            .parse::<u8>()
            .map_err(|_| invalid_table("automatic-model order exceeds u8".to_owned()))
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), OracleError> {
        if self.text.as_bytes().get(self.position) != Some(&expected) {
            return Err(invalid_table(format!(
                "automatic-model description expected {:?}",
                char::from(expected)
            )));
        }
        self.position += 1;
        Ok(())
    }

    fn skip_whitespace(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.position)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.position += 1;
        }
    }

    fn next_is_whitespace(&self) -> bool {
        self.text
            .as_bytes()
            .get(self.position)
            .is_some_and(u8::is_ascii_whitespace)
    }

    fn is_at_end(&self) -> bool {
        self.position == self.text.len()
    }
}
