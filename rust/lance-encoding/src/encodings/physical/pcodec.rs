// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Pcodec Miniblock Encoding
//!
//! Pcodec is a lossless codec for numerical sequences that typically gets better
//! compression ratio than alternatives. It uses a holistic 3-step approach:
//! modes, delta encoding, and binning.
//!
//! ## Supported Types
//!
//! - 16-bit: u16, i16, f16
//! - 32-bit: u32, i32, f32
//! - 64-bit: u64, i64, f64
//!
//! ## How It Works
//!
//! Pcodec compresses numerical data by:
//! 1. **Modes**: Identifying approximate structure (e.g., multiples of a constant)
//! 2. **Delta encoding**: Using differences between consecutive values when beneficial
//! 3. **Binning**: Representing values as entropy-coded bins with exact offsets
//!
//! ## Chunk Handling
//!
//! - Pcodec compresses entire pages at once (not individual mini-block chunks)
//! - This is because Pcodec needs to see the full sequence for optimal compression
//! - Random access is not supported within a compressed page

use std::fmt::Debug;

use crate::buffer::LanceBuffer;
use crate::compression::MiniBlockDecompressor;
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor,
};
use crate::format::pb21::CompressiveEncoding;
use lance_core::Result;
use pco::data_types::Number;
use pco::{ChunkConfig, PagingSpec};
use snafu::location;

/// Default compression level for Pcodec (range: 0-12)
const DEFAULT_COMPRESSION_LEVEL: usize = 8;

/// Maximum chunk size (8KiB - header overhead)
const MAX_MINIBLOCK_BYTES: usize = 8 * 1024 - 6;

/// Pcodec encoder for fixed-width numerical values
#[derive(Debug, Clone)]
pub struct PcodecMiniBlockEncoder {
    bits_per_value: usize,
    compression_level: usize,
}

impl PcodecMiniBlockEncoder {
    pub fn new(bits_per_value: usize) -> Self {
        Self::with_compression_level(bits_per_value, DEFAULT_COMPRESSION_LEVEL)
    }

    pub fn with_compression_level(bits_per_value: usize, compression_level: usize) -> Self {
        debug_assert!(
            bits_per_value == 16 || bits_per_value == 32 || bits_per_value == 64,
            "Pcodec only supports 16, 32, or 64 bit values"
        );
        Self {
            bits_per_value,
            compression_level: compression_level.min(12),
        }
    }
}

fn pcodec_encode<T: Number + bytemuck::Pod>(
    data: &[u8],
    num_values: u64,
    compression_level: usize,
) -> Result<Vec<u8>> {
    let values: &[T] = bytemuck::cast_slice(data);
    debug_assert_eq!(values.len(), num_values as usize);

    let config = ChunkConfig::default()
        .with_compression_level(compression_level)
        .with_paging_spec(PagingSpec::EqualPagesUpTo(MAX_MINIBLOCK_BYTES));

    pco::standalone::simple_compress(values, &config).map_err(|e| lance_core::Error::InvalidInput {
        source: format!("Pcodec compression failed: {}", e).into(),
        location: location!(),
    })
}

fn pcodec_decode<T: Number + bytemuck::Pod>(compressed: &[u8], num_values: u64) -> Result<Vec<u8>> {
    let values: Vec<T> =
        pco::standalone::simple_decompress(compressed).map_err(|e| lance_core::Error::InvalidInput {
            source: format!("Pcodec decompression failed: {}", e).into(),
            location: location!(),
        })?;

    if values.len() != num_values as usize {
        return Err(lance_core::Error::InvalidInput {
            source: format!(
                "Pcodec decompression expected {} values but got {}",
                num_values,
                values.len()
            )
            .into(),
            location: location!(),
        });
    }

    Ok(bytemuck::cast_slice(&values).to_vec())
}

impl MiniBlockCompressor for PcodecMiniBlockEncoder {
    fn compress(&self, page: DataBlock) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match page {
            DataBlock::FixedWidth(data) => {
                let num_values = data.num_values;

                if num_values == 0 {
                    return Ok((
                        MiniBlockCompressed {
                            data: vec![],
                            chunks: vec![],
                            num_values: 0,
                        },
                        crate::format::pb21::CompressiveEncoding {
                            compression: Some(
                                crate::format::pb21::compressive_encoding::Compression::Pcodec(
                                    crate::format::pb21::Pcodec {
                                        bits_per_value: self.bits_per_value as u64,
                                        compression_level: Some(self.compression_level as u32),
                                    },
                                ),
                            ),
                        },
                    ));
                }

                let data_slice = data.data.as_ref();

                // Compress the entire page using the appropriate type
                let compressed = match self.bits_per_value {
                    16 => pcodec_encode::<u16>(data_slice, num_values, self.compression_level)?,
                    32 => pcodec_encode::<u32>(data_slice, num_values, self.compression_level)?,
                    64 => pcodec_encode::<u64>(data_slice, num_values, self.compression_level)?,
                    _ => {
                        return Err(lance_core::Error::InvalidInput {
                            source: format!(
                                "Pcodec encoding only supports 16, 32, or 64 bit values, got {}",
                                self.bits_per_value
                            )
                            .into(),
                            location: location!(),
                        })
                    }
                };

                // Create a single chunk containing all the compressed data
                let chunk = MiniBlockChunk {
                    buffer_sizes: vec![compressed.len() as u32],
                    log_num_values: 0, // Last chunk (contains all values)
                };

                let encoding = crate::format::pb21::CompressiveEncoding {
                    compression: Some(
                        crate::format::pb21::compressive_encoding::Compression::Pcodec(
                            crate::format::pb21::Pcodec {
                                bits_per_value: self.bits_per_value as u64,
                                compression_level: Some(self.compression_level as u32),
                            },
                        ),
                    ),
                };

                Ok((
                    MiniBlockCompressed {
                        data: vec![LanceBuffer::from(compressed)],
                        chunks: vec![chunk],
                        num_values,
                    },
                    encoding,
                ))
            }
            _ => Err(lance_core::Error::InvalidInput {
                source: "Pcodec encoding only supports FixedWidth data blocks".into(),
                location: location!(),
            }),
        }
    }
}

/// Pcodec decompressor
#[derive(Debug)]
pub struct PcodecMiniBlockDecompressor {
    bits_per_value: usize,
}

impl PcodecMiniBlockDecompressor {
    pub fn new(bits_per_value: usize) -> Self {
        debug_assert!(
            bits_per_value == 16 || bits_per_value == 32 || bits_per_value == 64,
            "Pcodec only supports 16, 32, or 64 bit values"
        );
        Self { bits_per_value }
    }

    fn bytes_per_value(&self) -> usize {
        self.bits_per_value / 8
    }
}

impl MiniBlockDecompressor for PcodecMiniBlockDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if num_values == 0 {
            return Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::empty(),
                bits_per_value: self.bits_per_value as u64,
                num_values: 0,
                block_info: BlockInfo::new(),
            }));
        }

        if data.len() != 1 {
            return Err(lance_core::Error::InvalidInput {
                source: format!(
                    "Pcodec decompression expects 1 buffer, but got {}",
                    data.len()
                )
                .into(),
                location: location!(),
            });
        }

        let compressed = &data[0];

        // Decompress using the appropriate type
        let decompressed = match self.bits_per_value {
            16 => pcodec_decode::<u16>(compressed.as_ref(), num_values)?,
            32 => pcodec_decode::<u32>(compressed.as_ref(), num_values)?,
            64 => pcodec_decode::<u64>(compressed.as_ref(), num_values)?,
            _ => {
                return Err(lance_core::Error::InvalidInput {
                    source: format!(
                        "Pcodec decoding only supports 16, 32, or 64 bit values, got {}",
                        self.bits_per_value
                    )
                    .into(),
                    location: location!(),
                })
            }
        };

        let expected_bytes = num_values as usize * self.bytes_per_value();
        if decompressed.len() != expected_bytes {
            return Err(lance_core::Error::InvalidInput {
                source: format!(
                    "Expected {} bytes after decompression, but got {}",
                    expected_bytes,
                    decompressed.len()
                )
                .into(),
                location: location!(),
            });
        }

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(decompressed),
            bits_per_value: self.bits_per_value as u64,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;

    fn test_round_trip<T: bytemuck::Pod + PartialEq + std::fmt::Debug>(
        data: Vec<T>,
        bits_per_value: usize,
    ) {
        let encoder = PcodecMiniBlockEncoder::new(bits_per_value);
        let decompressor = PcodecMiniBlockDecompressor::new(bits_per_value);

        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: bits_per_value as u64,
            num_values: data.len() as u64,
            block_info: BlockInfo::new(),
        });

        let (compressed, _encoding) = encoder.compress(data_block).unwrap();

        let decompressed = decompressor
            .decompress(compressed.data, compressed.num_values)
            .unwrap();

        let DataBlock::FixedWidth(decompressed_block) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        let result: &[T] = bytemuck::cast_slice(decompressed_block.data.as_ref());
        assert_eq!(result, &data[..]);
    }

    #[test]
    fn test_round_trip_u16() {
        let data: Vec<u16> = (0..1000).collect();
        test_round_trip(data, 16);
    }

    #[test]
    fn test_round_trip_i32() {
        let data: Vec<i32> = (-500..500).collect();
        test_round_trip(data, 32);
    }

    #[test]
    fn test_round_trip_u32() {
        let data: Vec<u32> = (0..1000).collect();
        test_round_trip(data, 32);
    }

    #[test]
    fn test_round_trip_i64() {
        let data: Vec<i64> = (-500..500).collect();
        test_round_trip(data, 64);
    }

    #[test]
    fn test_round_trip_f32() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        test_round_trip(data, 32);
    }

    #[test]
    fn test_round_trip_f64() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        test_round_trip(data, 64);
    }

    #[test]
    fn test_empty_data() {
        let encoder = PcodecMiniBlockEncoder::new(32);
        let decompressor = PcodecMiniBlockDecompressor::new(32);

        let data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::empty(),
            bits_per_value: 32,
            num_values: 0,
            block_info: BlockInfo::new(),
        });

        let (compressed, _encoding) = encoder.compress(data_block).unwrap();

        let decompressed = decompressor.decompress(compressed.data, 0).unwrap();
        let DataBlock::FixedWidth(decompressed_block) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        assert_eq!(decompressed_block.num_values, 0);
        assert_eq!(decompressed_block.data.len(), 0);
    }

    #[test]
    fn test_compression_level() {
        // Test with different compression levels
        for level in [0, 4, 8, 12] {
            let encoder = PcodecMiniBlockEncoder::with_compression_level(32, level);
            let decompressor = PcodecMiniBlockDecompressor::new(32);

            let data: Vec<i32> = (0..1000).collect();
            let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

            let data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::from(bytes),
                bits_per_value: 32,
                num_values: data.len() as u64,
                block_info: BlockInfo::new(),
            });

            let (compressed, _encoding) = encoder.compress(data_block).unwrap();

            let decompressed = decompressor
                .decompress(compressed.data, compressed.num_values)
                .unwrap();

            let DataBlock::FixedWidth(decompressed_block) = &decompressed else {
                panic!("Expected FixedWidth DataBlock")
            };

            let result: &[i32] = bytemuck::cast_slice(decompressed_block.data.as_ref());
            assert_eq!(result, &data[..]);
        }
    }

    #[test]
    fn test_timestamp_like_data() {
        // Timestamps are a common use case for Pcodec
        // Simulate sequential timestamps with small deltas
        let base: i64 = 1704067200000000000; // Some nanosecond timestamp
        let data: Vec<i64> = (0..10000).map(|i| base + i * 1000000).collect();

        let encoder = PcodecMiniBlockEncoder::new(64);
        let decompressor = PcodecMiniBlockDecompressor::new(64);

        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let original_size = bytes.len();

        let data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: data.len() as u64,
            block_info: BlockInfo::new(),
        });

        let (compressed, _encoding) = encoder.compress(data_block).unwrap();
        let compressed_size: usize = compressed.data.iter().map(|b| b.len()).sum();

        // Pcodec should achieve good compression on sequential data
        assert!(
            compressed_size < original_size,
            "Compressed size {} should be less than original {}",
            compressed_size,
            original_size
        );

        let decompressed = decompressor
            .decompress(compressed.data, compressed.num_values)
            .unwrap();

        let DataBlock::FixedWidth(decompressed_block) = &decompressed else {
            panic!("Expected FixedWidth DataBlock")
        };

        let result: &[i64] = bytemuck::cast_slice(decompressed_block.data.as_ref());
        assert_eq!(result, &data[..]);
    }

    #[test]
    fn test_from_arrow_arrays() {
        // Test with Arrow array types
        let i32_arr = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let data_block = DataBlock::from_array(i32_arr.clone());

        let encoder = PcodecMiniBlockEncoder::new(32);
        let decompressor = PcodecMiniBlockDecompressor::new(32);

        let (compressed, _) = encoder.compress(data_block).unwrap();
        let decompressed = decompressor
            .decompress(compressed.data, compressed.num_values)
            .unwrap();

        let DataBlock::FixedWidth(block) = decompressed else {
            panic!("Expected FixedWidth")
        };

        let result: &[i32] = bytemuck::cast_slice(block.data.as_ref());
        assert_eq!(result.len(), 10);
    }
}
