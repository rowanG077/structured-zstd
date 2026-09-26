use super::{Args, decode};
use alloc::{vec, vec::Vec};

// Intentional modular arithmetic supplies reproducible data without depending
// on the encoder or table builder that the kernel is meant to accelerate.
fn random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state ^ (*state >> 31)
}

fn load(source: &[u8], position: usize) -> u64 {
    u64::from_le_bytes(source[position..position + 8].try_into().unwrap())
}

fn compare_with_scalar<const FIXED_PAIR: bool, const UNIFORM: u8>() {
    if !std::is_x86_feature_detected!("bmi2") {
        return;
    }
    const GUARD: usize = 16;
    let mut seed = 1234;
    for case in 0..5000 {
        let iterations = case % 40 + 1;
        // Exactly the input/output budgets in decode's safety contract. The
        // First 80 variable-bit cases force maximum consumption and reach
        // input offset zero; every variant covers sentinel positions 0..=8.
        let input_start = iterations * 7;
        let maximal_consumption = case < 80;
        let mut table = [0u32; 2048];
        // The oracle keeps logical values independently of the packed layout.
        let mut logical_table = [(0u16, 0u32, 0usize); 2048];
        for (entry, logical) in table.iter_mut().zip(&mut logical_table) {
            let value = random(&mut seed);
            let bits = if UNIFORM != 0 {
                u32::from(UNIFORM)
            } else if maximal_consumption {
                11
            } else {
                ((value >> 32) % 11 + 1) as u32
            };
            let length = if FIXED_PAIR {
                2
            } else if maximal_consumption {
                (case % 2 + 1) as u32
            } else {
                ((value >> 48) % 2 + 1) as u32
            };
            *logical = (value as u16, bits, length as usize);
            *entry = (value as u32 & 0xffff) | (bits << 16) | (length << 22);
        }
        let source: [Vec<u8>; 4] = core::array::from_fn(|_| {
            (0..input_start + 8)
                .map(|_| random(&mut seed) as u8)
                .collect()
        });
        let mut output: [Vec<u8>; 4] =
            core::array::from_fn(|_| vec![0xa5; GUARD + iterations * 10 + GUARD]);
        let mut expected = output.clone();
        let bits = core::array::from_fn(|stream| {
            let sentinel = if maximal_consumption {
                8
            } else {
                (case / 40 + stream) % 9
            };
            (load(&source[stream], input_start) | 1) << sentinel
        });
        let mut args = Args {
            ip: core::array::from_fn(|i| source[i].as_ptr().wrapping_add(input_start)),
            op: core::array::from_fn(|i| output[i].as_mut_ptr().wrapping_add(GUARD)),
            bits,
            table: table.as_ptr(),
            iters: iterations,
        };
        let mut input_positions = [input_start; 4];
        let mut output_positions = [GUARD; 4];
        let mut scalar_bits = bits;
        for _ in 0..iterations {
            for stream in 0..4 {
                for _ in 0..5 {
                    let (symbols, consumed_bits, length) =
                        logical_table[(scalar_bits[stream] >> 53) as usize];
                    let position = output_positions[stream];
                    expected[stream][position..position + 2]
                        .copy_from_slice(&symbols.to_le_bytes());
                    scalar_bits[stream] <<= consumed_bits;
                    output_positions[stream] += length;
                }
                let consumed = scalar_bits[stream].trailing_zeros() as usize;
                assert!(consumed <= 63, "the sentinel must survive all five lookups");
                input_positions[stream] -= consumed / 8;
                scalar_bits[stream] =
                    (load(&source[stream], input_positions[stream]) | 1) << (consumed % 8);
            }
        }
        // SAFETY: BMI2 was detected above. Every entry consumes 1..=11 bits
        // and emits 1..=2 bytes (always 2 for FIXED_PAIR). Each source provides
        // iterations*7 bytes before ip and 8 at ip, each output provides
        // iterations*10 bytes at op, and initial sentinels occupy bits 0..=8.
        // Uniform variants use fixed pairs with exactly UNIFORM bits per entry.
        unsafe { decode::<FIXED_PAIR, UNIFORM>(&mut args) };
        for stream in 0..4 {
            let context = (FIXED_PAIR, UNIFORM, case, stream);
            assert_eq!(
                args.ip[stream],
                source[stream]
                    .as_ptr()
                    .wrapping_add(input_positions[stream]),
                "input cursor: {context:?}"
            );
            assert_eq!(
                args.op[stream],
                output[stream]
                    .as_ptr()
                    .wrapping_add(output_positions[stream])
                    .cast_mut(),
                "output cursor: {context:?}"
            );
            assert_eq!(args.bits[stream], scalar_bits[stream], "bits: {context:?}");
            // Compare the whole allocation, including both canaries and the
            // untouched space after variable-length output.
            assert_eq!(output[stream], expected[stream], "output: {context:?}");
        }
    }
}

#[test]
fn mixed_entries_match_scalar_and_preserve_canaries() {
    compare_with_scalar::<false, 0>();
}

#[test]
fn fixed_pairs_match_scalar_and_preserve_canaries() {
    compare_with_scalar::<true, 0>();
}

#[test]
fn uniform_four_bit_codes_match_scalar_and_preserve_canaries() {
    compare_with_scalar::<true, 8>();
}

#[test]
fn uniform_five_bit_codes_match_scalar_and_preserve_canaries() {
    compare_with_scalar::<true, 10>();
}

#[test]
fn packed_length_byte_is_portable_between_endian_layouts() {
    for symbols in [0u16, 0x00ff, 0xff00, 0x5aa5, 0xffff] {
        for bits in 1u8..=11 {
            for length in 1u8..=2 {
                let code_len = bits | (length << 6);
                let entry = u32::from(symbols) | (u32::from(code_len) << 16);
                let little = entry.to_le_bytes();
                let big = entry.to_be_bytes();
                assert_eq!(little[2], code_len);
                assert_eq!(big[1], code_len);
                assert_eq!(little[3], 0);
                assert_eq!(big[0], 0);
                assert_eq!(u16::from_le_bytes([little[0], little[1]]), symbols);
                assert_eq!(u16::from_be_bytes([big[2], big[3]]), symbols);
                assert_eq!(code_len & 63, bits);
                assert_eq!(code_len >> 6, length);
                assert_eq!(
                    0x93e156c7a82db409u64.wrapping_shl(u32::from(code_len)),
                    0x93e156c7a82db409u64 << bits
                );
            }
        }
    }
}
