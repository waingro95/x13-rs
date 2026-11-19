use std::slice::ChunksExact;

/// A dense matrix stored row by row: element `(row, column)` sits at
/// `row * column_count + column`.
///
/// Build 62 keeps its observation-by-variable work arrays this way (`[X:y]`,
/// outlier candidate columns, the exact-filter cross terms), so each row is a
/// contiguous slice.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RowMajorMatrix {
    values: Vec<f64>,
    columns: usize,
}

impl RowMajorMatrix {
    /// # Panics
    ///
    /// Panics if `columns` is zero or `values` does not fill whole rows.
    pub(crate) fn new(values: Vec<f64>, columns: usize) -> Self {
        debug_assert!(columns > 0, "a row-major matrix needs at least one column");
        // Row iteration would otherwise silently drop a partial row.
        assert_eq!(
            values.len() % columns,
            0,
            "row-major values must fill whole rows"
        );
        Self { values, columns }
    }

    pub(crate) fn row_count(&self) -> usize {
        self.values.len() / self.columns
    }

    pub(crate) const fn column_count(&self) -> usize {
        self.columns
    }

    pub(crate) fn rows(&self) -> ChunksExact<'_, f64> {
        self.values.chunks_exact(self.columns)
    }

    pub(crate) fn as_slice(&self) -> &[f64] {
        &self.values
    }

    pub(crate) fn into_values(self) -> Vec<f64> {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::RowMajorMatrix;

    #[test]
    fn rows_are_contiguous_slices() {
        let matrix = RowMajorMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        assert_eq!((matrix.row_count(), matrix.column_count()), (2, 3));
        assert_eq!(
            matrix.rows().collect::<Vec<_>>(),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        );
        assert_eq!(matrix.as_slice(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let empty = RowMajorMatrix::new(Vec::new(), 2);
        assert_eq!(empty.row_count(), 0);
        assert_eq!(empty.rows().count(), 0);
    }

    #[test]
    // Debug builds stop at the assertion, release builds at the remainder by
    // zero.
    #[should_panic]
    fn zero_columns_are_rejected() {
        let _ = RowMajorMatrix::new(vec![1.0, 2.0], 0);
    }

    #[test]
    #[should_panic(expected = "whole rows")]
    fn partial_rows_are_rejected() {
        let _ = RowMajorMatrix::new(vec![1.0, 2.0], 3);
    }
}
