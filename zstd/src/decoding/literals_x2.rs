/*
BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook, nor Meta, nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//! Four-stream Huffman payload decoding with one or two symbols per lookup.
//! Input/output budgets are established by the caller before each chunk.

pub(super) struct Args {
    pub ip: [*const u8; 4],
    pub op: [*mut u8; 4],
    pub bits: [u64; 4],
    pub table: *const u32,
    pub iters: usize,
}

/// # Safety
/// The caller has resolved BMI2 on x86-64. `iters > 0`; each input pointer has
/// `iters * 7` readable preceding bytes and eight readable bytes at its current
/// position. Each output pointer has `iters * 10` writable bytes. The table
/// contains 2048 initialized entries with code lengths 1..=11 and output counts
/// 1 or 2. Symbols occupy the low 16 bits; bits 16..=23 contain
/// `code_bits | (output_count << 6)` and the high byte is zero. Stream sentinels
/// occupy positions 0..=8. `FIXED_PAIR` means every
/// table entry emits two symbols; nonzero `UNIFORM` additionally guarantees
/// every entry consumes exactly that many bits.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "bmi2"))]
pub(super) unsafe fn decode<const FIXED_PAIR: bool, const UNIFORM: u8>(args: &mut Args) {
    debug_assert!(args.iters > 0);
    debug_assert!(UNIFORM <= 11 && (UNIFORM == 0 || FIXED_PAIR));
    let mut b0 = args.bits[0];
    let mut b1 = args.bits[1];
    let mut b2 = args.bits[2];
    let mut b3 = args.bits[3];
    let mut ip0 = args.ip[0];
    let mut ip1 = args.ip[1];
    let mut ip2 = args.ip[2];
    let mut ip3 = args.ip[3];
    let mut op0 = args.op[0];
    let mut op1 = args.op[1];
    let mut op2 = args.op[2];
    let mut op3 = args.op[3];
    let table = args.table;

    macro_rules! emit {
        ($bits:ident, $output:ident, $offset:literal) => {{
            // The top eleven bits index exactly the 2048-entry table.
            let entry_ptr = unsafe { table.add(($bits >> 53) as usize) };
            if FIXED_PAIR {
                let entry = unsafe { entry_ptr.read() };
                let output = unsafe { $output.add(2 * $offset) };
                unsafe { output.cast::<u16>().write_unaligned((entry as u16).to_le()) };
                $bits = $bits.wrapping_shl(if UNIFORM == 0 {
                    entry >> 16
                } else {
                    u32::from(UNIFORM)
                });
            } else {
                let bytes = entry_ptr.cast::<u8>();
                let symbols_offset = if cfg!(target_endian = "little") { 0 } else { 2 };
                let bits_offset = if cfg!(target_endian = "little") { 2 } else { 1 };
                let symbols = unsafe { bytes.add(symbols_offset).cast::<u16>().read_unaligned() };
                let code_len = unsafe { bytes.add(bits_offset).read() };
                unsafe { $output.cast::<u16>().write_unaligned(symbols.to_le()) };
                // u64 wrapping shifts use the low six count bits, leaving the
                // two output-length bits available without another table load.
                $bits = $bits.wrapping_shl(u32::from(code_len));
                $output = unsafe { $output.add(usize::from(code_len >> 6)) };
            }
        }};
    }
    macro_rules! round {
        ($offset:literal) => {
            emit!(b0, op0, $offset);
            emit!(b1, op1, $offset);
            emit!(b2, op2, $offset);
            emit!(b3, op3, $offset);
        };
    }
    macro_rules! reload {
        ($bits:ident, $input:ident) => {{
            // Five entries consume at most 55 bits. The sentinel consequently
            // remains set, and at most seven input bytes are consumed.
            let consumed = $bits.trailing_zeros();
            $input = unsafe { $input.sub((consumed >> 3) as usize) };
            let window = u64::from_le(unsafe { $input.cast::<u64>().read_unaligned() });
            $bits = (window | 1) << (consumed & 7);
        }};
    }
    for _ in 0..args.iters {
        round!(0);
        round!(1);
        round!(2);
        round!(3);
        round!(4);
        if FIXED_PAIR {
            op0 = unsafe { op0.add(10) };
            op1 = unsafe { op1.add(10) };
            op2 = unsafe { op2.add(10) };
            op3 = unsafe { op3.add(10) };
        }
        reload!(b0, ip0);
        reload!(b1, ip1);
        reload!(b2, ip2);
        reload!(b3, ip3);
    }
    args.ip = [ip0, ip1, ip2, ip3];
    args.op = [op0, op1, op2, op3];
    args.bits = [b0, b1, b2, b3];
}

#[cfg(all(test, feature = "std"))]
#[path = "literals_x2/tests.rs"]
mod tests;
