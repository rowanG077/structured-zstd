/// Update the most recently used offsets to reflect the provided offset value, and return the
/// "actual" offset needed because offsets are not stored in a raw way, some transformations are needed
/// before you get a functional number.
pub(crate) fn do_offset_history(offset_value: u32, lit_len: u32, scratch: &mut [u32; 3]) -> u32 {
    // Fast path: offset_value >= 4 means a fresh (non-repcode) offset, which
    // is the dominant case for non-trivial corpora. Upstream zstd (zstd_decompress_block.c
    // ZSTD_updateRep) special-cases this with a straight shift: rotate the
    // history down and store `offset_value - 3` at slot 0. No rule table, no
    // branchless masks. Repcodes 1..=3 select from the offset history below.
    if offset_value >= 4 {
        let actual = offset_value - 3;
        scratch[2] = scratch[1];
        scratch[1] = scratch[0];
        scratch[0] = actual;
        return actual;
    }

    do_offset_history_repcode(offset_value, lit_len, scratch)
}

// Repcode values select the last three offsets; a zero literal length shifts
// the selection by one, with the fourth choice meaning the last offset minus
// one. Keep the selected offset most recent, without moving older entries when
// the first or second one was selected.
#[inline]
fn do_offset_history_repcode(offset_value: u32, lit_len: u32, scratch: &mut [u32; 3]) -> u32 {
    debug_assert!(offset_value < 4);
    if offset_value == 0 {
        return 0;
    }
    let index = offset_value - 1 + u32::from(lit_len == 0);
    if index == 0 {
        return scratch[0];
    }
    let actual = if index == 3 {
        scratch[0].wrapping_sub(1)
    } else {
        scratch[index as usize]
    };
    if index != 1 {
        scratch[2] = scratch[1];
    }
    scratch[1] = scratch[0];
    scratch[0] = actual;
    actual
}

#[cfg(test)]
mod tests;
