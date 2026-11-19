//! The `[X:y]` matrix a regression is solved from.

use crate::Error;
use crate::numerics::RowMajorMatrix;

/// Build 62's normalized `[X:y]` matrix: one row per observation, the
/// regressors in caller order and the response in the last column.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionMatrix {
    matrix: RowMajorMatrix,
}

impl RegressionMatrix {
    /// Wraps `values`, stored row by row with the response last in each row.
    pub(crate) fn from_row_major(values: Vec<f64>, row_count: usize, column_count: usize) -> Self {
        debug_assert!(
            column_count > 0,
            "a regression matrix needs a response column"
        );
        debug_assert_eq!(
            row_count.checked_mul(column_count),
            Some(values.len()),
            "regression matrix dimensions must match its values"
        );
        Self::from(RowMajorMatrix::new(values, column_count))
    }

    /// A matrix whose only column is the response.
    pub(crate) fn without_regressors(response: &[f64]) -> Self {
        Self::from_row_major(response.to_vec(), response.len(), 1)
    }

    /// Interleaves `columns` and `response` into rows. Every column must be
    /// as long as the response.
    pub(crate) fn from_columns<C: AsRef<[f64]>>(
        columns: &[C],
        response: &[f64],
    ) -> Result<Self, Error> {
        if columns
            .iter()
            .any(|column| column.as_ref().len() != response.len())
        {
            return Err(Error::internal(
                "regressor column length differs from the response",
            ));
        }
        let column_count = columns.len() + 1;
        let mut values = Vec::with_capacity(response.len() * column_count);
        for (row, response) in response.iter().copied().enumerate() {
            for column in columns {
                values.push(column.as_ref()[row]);
            }
            values.push(response);
        }
        Ok(Self::from_row_major(values, response.len(), column_count))
    }

    pub(crate) const fn as_matrix(&self) -> &RowMajorMatrix {
        &self.matrix
    }

    pub(crate) fn into_matrix(self) -> RowMajorMatrix {
        self.matrix
    }

    pub(crate) fn values(&self) -> &[f64] {
        self.matrix.as_slice()
    }

    pub(crate) fn row_count(&self) -> usize {
        self.matrix.row_count()
    }

    pub(crate) const fn column_count(&self) -> usize {
        self.matrix.column_count()
    }
}

/// The last column of any row-major matrix is read as the response.
impl From<RowMajorMatrix> for RegressionMatrix {
    fn from(matrix: RowMajorMatrix) -> Self {
        Self { matrix }
    }
}

#[cfg(test)]
mod tests {
    use super::RegressionMatrix;

    #[test]
    fn columns_are_stored_in_caller_order_with_response_last() {
        let response = [10.0, 20.0, 30.0];
        let columns = [vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let matrix = RegressionMatrix::from_columns(&columns, &response).unwrap();

        assert_eq!(matrix.row_count(), 3);
        assert_eq!(matrix.column_count(), 3);
        assert_eq!(
            matrix.values(),
            [1.0, 4.0, 10.0, 2.0, 5.0, 20.0, 3.0, 6.0, 30.0]
        );
    }

    #[test]
    fn a_column_of_another_length_is_an_error() {
        assert!(matches!(
            RegressionMatrix::from_columns(&[vec![1.0]], &[1.0, 2.0]),
            Err(crate::Error::Internal(_))
        ));
    }
}
