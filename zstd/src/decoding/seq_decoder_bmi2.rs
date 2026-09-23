//! BMI2 entry point for the shared sequence decoder.
//!
//! Runtime dispatch confirms BMI2 support before entering this function. The
//! inlined implementation then uses BMI2 bit masks with the same sequence
//! validation, output bounds, and rollback behavior as the scalar entry point.

#![cfg(target_arch = "x86_64")]

use super::buffer_backend::BufferBackend;
use super::decode_buffer::DecodeBuffer;
use super::scratch::FSEScratch;
use crate::blocks::sequence_section::SequencesHeader;
use crate::cpu_kernel::Bmi2Kernel;
use crate::decoding::errors::DecompressBlockError;

/// Decode sequences using BMI2 bit masks.
///
/// # Safety
/// The caller must have verified BMI2 availability on the running CPU.
#[target_feature(enable = "bmi2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn decode_and_execute_sequences_bmi2<'fse, B: BufferBackend>(
    section: &SequencesHeader,
    source: &[u8],
    fse: &'fse mut FSEScratch,
    buffer: &mut DecodeBuffer<B>,
    offset_hist: &mut [u32; 3],
    literals_buffer: &[u8],
    literals_len: usize,
    dict: Option<&'fse crate::decoding::dictionary::Dictionary>,
) -> Result<(), DecompressBlockError> {
    super::seq_decoder_scalar::decode_and_execute_sequences_impl::<B, Bmi2Kernel>(
        section,
        source,
        fse,
        buffer,
        offset_hist,
        literals_buffer,
        literals_len,
        dict,
    )
}
