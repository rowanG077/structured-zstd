use super::*;
use crate::decoding::dictionary::Dictionary;
use crate::decoding::flat_buf::FlatBuf;
use crate::decoding::user_slice_buf::UserSliceBackend;
use alloc::{vec, vec::Vec};

fn available() -> bool {
    std::is_x86_feature_detected!("bmi2") && std::is_x86_feature_detected!("avx2")
}

// Exercise the actual execute macro with one cursor carried across sequences,
// independently of the entropy encoder. The public decoder's ordinary tests
// separately cover its entropy parsing and rollback integration.
#[target_feature(enable = "bmi2,avx2")]
unsafe fn execute<B: BufferBackend>(
    buffer: &mut DecodeBuffer<B>,
    literals: &[u8],
    literal_len: usize,
    sequences: &[(u32, u32, u32)],
    dict: Option<&Dictionary>,
) -> Result<(), DecompressBlockError> {
    let dict_content = dict.map_or(&[][..], |d| d.dict_content.as_slice());
    let mut cursor = OutCursor::capture(buffer);
    let mut lit_cur: usize = 0;
    for &(ll, ml, offset) in sequences {
        let result = execute_one_body!(
            &mut cursor,
            buffer,
            dict,
            dict_content,
            literals,
            &mut lit_cur,
            literal_len,
            ll,
            ml,
            offset
        );
        if let Err(error) = result {
            cursor.publish(buffer);
            return Err(error);
        }
    }
    cursor.publish(buffer);
    Ok(())
}

#[test]
fn pointer_cursor_live_prefix_and_exact_tail_boundaries() {
    if !available() {
        return;
    }
    // A dropped prefix makes physical tail and live-prefix length differ.
    // In particular, head=35/live<16 must take the conservative overlap head.
    for head in [0, 5, 35] {
        for live in [0, 1, 7, 15, 16, 17, 32] {
            for ll in [0, 1, 15, 16, 17, 31, 32, 33] {
                let literals: Vec<u8> = (0..ll + crate::WILDCOPY_OVERLENGTH)
                    .map(|i| (i * 19 + 7) as u8)
                    .collect();
                for offset in [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33] {
                    if offset > live + ll {
                        continue;
                    }
                    for ml in [3, 8, 9, 16, 17, 24, 25, 32, 33, 65, 1025] {
                        let prefix: Vec<u8> =
                            (0..head + live).map(|i| (i * 13 + 5) as u8).collect();
                        let mut expected = prefix[head..].to_vec();
                        expected.extend_from_slice(&literals[..ll]);
                        for _ in 0..ml {
                            expected.push(expected[expected.len() - offset]);
                        }
                        let sequences = [(ll as u32, ml as u32, offset as u32)];
                        for slack in [0, 1, 15, 30, 31, 32] {
                            let limit = head + expected.len() + slack;
                            let mut out = vec![0xCD; limit + 32];
                            {
                                let backend = UserSliceBackend::from_slice(&mut out[..limit]);
                                let mut buffer = DecodeBuffer::from_backend(backend, 4096);
                                buffer.try_push(&prefix).unwrap();
                                buffer.buffer_mut().drop_first_n(head);
                                unsafe {
                                    execute(&mut buffer, &literals, ll, &sequences, None).unwrap();
                                }
                                assert_eq!(buffer.buffer_ref().as_slices().0, expected);
                                assert_eq!(buffer.buffer_ref().tail(), head + expected.len());
                            }
                            assert!(out[limit..].iter().all(|&b| b == 0xCD));

                            // FlatBuf's spare capacity stays uninitialized.
                            // Limit writes separately to cover exact per-block
                            // ceilings even when the allocation has more room.
                            let backend = FlatBuf::with_capacity(limit);
                            let mut buffer = DecodeBuffer::from_backend(backend, 4096);
                            buffer.try_push(&prefix).unwrap();
                            buffer.buffer_mut().drop_first_n(head);
                            buffer.set_block_output_ceiling(ll + ml + slack);
                            unsafe {
                                execute(&mut buffer, &literals, ll, &sequences, None).unwrap();
                            }
                            assert_eq!(buffer.buffer_ref().as_slices().0, expected);
                            assert_eq!(buffer.buffer_ref().tail(), head + expected.len());
                            #[cfg(feature = "lsm")]
                            assert_eq!(buffer.total_output(), (prefix.len() + ll + ml) as u64);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn pointer_cursor_rejects_malformed_lengths_and_restores_checkpoint() {
    if !available() {
        return;
    }
    let literals = [b'x'; 1 + crate::WILDCOPY_OVERLENGTH];
    for sequences in [
        [(0, 3, 1), (u32::MAX, 3, 1)],
        [(0, 3, 1), (0, u32::MAX, 1)],
        [(0, 3, 1), (0, 3, 0)],
        [(0, 3, 1), (0, 3, u32::MAX)],
    ] {
        let mut out = [0xCD; 96];
        let mut buffer =
            DecodeBuffer::from_backend(UserSliceBackend::from_slice(&mut out[..64]), 64);
        buffer.try_push(b"prefix").unwrap();
        let checkpoint = buffer.checkpoint();
        let result = unsafe { execute(&mut buffer, &literals, 1, &sequences, None) };
        assert!(result.is_err());
        assert_eq!(buffer.buffer_ref().tail(), 9);
        assert!(buffer.try_restore_checkpoint(checkpoint));
        assert_eq!(buffer.buffer_ref().as_slices().0, b"prefix");
        // The same backing buffer remains usable after rollback.
        unsafe { execute(&mut buffer, &literals, 1, &[(1, 3, 1)], None).unwrap() };
        assert_eq!(buffer.buffer_ref().as_slices().0, b"prefixxxxx");
        drop(buffer);
        assert!(out[64..].iter().all(|&b| b == 0xCD));
    }
}

#[test]
fn pointer_cursor_recaptures_after_dictionary_growth_and_tracks_output() {
    if !available() {
        return;
    }
    let dict = Dictionary::from_raw_content(0, b"abc".to_vec()).unwrap();
    let literals = [0; crate::WILDCOPY_OVERLENGTH];
    let mut buffer = DecodeBuffer::<FlatBuf>::new(4096);
    assert_eq!(buffer.buffer_ref().cap(), 0);
    // The first match crosses the dictionary/output boundary, requiring the
    // cold backend path and allocating from an initially dangling Vec pointer.
    // The next sequence must use the new allocation, with the carried pointer
    // still unpublished until the end of this execute call.
    unsafe {
        execute(
            &mut buffer,
            &literals,
            0,
            &[(0, 8, 3), (0, 3, 3)],
            Some(&dict),
        )
        .unwrap();
    }
    assert_eq!(buffer.buffer_ref().as_slices().0, b"abcabcabcab");
    #[cfg(feature = "lsm")]
    assert_eq!(buffer.total_output(), 11);
    let checkpoint = buffer.checkpoint();
    let result = unsafe {
        execute(
            &mut buffer,
            &literals,
            0,
            &[(0, 3, 3), (0, 3, 0)],
            Some(&dict),
        )
    };
    assert!(result.is_err());
    assert!(buffer.try_restore_checkpoint(checkpoint));
    assert_eq!(buffer.buffer_ref().as_slices().0, b"abcabcabcab");
    #[cfg(feature = "lsm")]
    assert_eq!(buffer.total_output(), 11);
}
