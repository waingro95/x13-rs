//! Build 62's table of Easter Sunday dates.
//!
//! Build 62 looks Easter up rather than computing it. The regARIMA Easter
//! regressor reads the table from `adestr.f` and the X-11 Easter adjustment
//! reads the same values from the first column of `kdate.prm`.

/// The first year of [`EASTER_DATE_OFFSETS`].
pub(crate) const FIRST_EASTER_YEAR: i32 = 1901;

/// Days from March 22 to Easter Sunday, for each year from 1901 through 2100.
pub(crate) const EASTER_DATE_OFFSETS: [u8; 200] = [
    16, 8, 21, 12, 32, 24, 9, 28, 20, 5, 25, 16, 1, 21, 13, 32, 17, 9, 29, 13, 5, 25, 10, 29, 21,
    13, 26, 17, 9, 29, 14, 5, 25, 10, 30, 21, 6, 26, 18, 2, 22, 14, 34, 18, 10, 30, 15, 6, 26, 18,
    3, 22, 14, 27, 19, 10, 30, 15, 7, 26, 11, 31, 23, 7, 27, 19, 4, 23, 15, 7, 20, 11, 31, 23, 8,
    27, 19, 4, 24, 15, 28, 20, 12, 31, 16, 8, 28, 12, 4, 24, 9, 28, 20, 12, 25, 16, 8, 21, 13, 32,
    24, 9, 29, 20, 5, 25, 17, 1, 21, 13, 33, 17, 9, 29, 14, 5, 25, 10, 30, 21, 13, 26, 18, 9, 29,
    14, 6, 25, 10, 30, 22, 6, 26, 18, 3, 22, 14, 34, 19, 10, 30, 15, 7, 26, 18, 3, 23, 14, 27, 19,
    11, 30, 15, 7, 27, 11, 31, 23, 8, 27, 19, 4, 24, 15, 7, 20, 12, 31, 23, 8, 28, 19, 4, 24, 16,
    28, 20, 12, 32, 16, 8, 28, 13, 4, 24, 9, 29, 20, 12, 25, 17, 8, 21, 13, 33, 24, 9, 29, 21, 6,
];
