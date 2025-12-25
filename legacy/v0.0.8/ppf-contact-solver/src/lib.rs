// Library entrypoint for reuse (e.g. in-process bindings).
// Keeps the existing binary behavior intact.

mod args;
mod backend;
mod builder;
mod bvh;
mod cvec;
mod cvecvec;
mod data;
mod mesh;
mod scene;
mod triutils;
mod inprocess;

extern crate nalgebra as na;

pub use args::{ProgramArgs, SimArgs};
pub use backend::{Backend, MeshSet};
pub use builder::Props;
pub use cvec::CVec;
pub use data::{BvhSet, DataSet, EdgeParam, EdgeProp, ParamSet};
pub use mesh::Mesh as SimMesh;
pub use scene::Scene;
pub use inprocess::InProcessSession;

use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use std::io::{Read, Write};

#[no_mangle]
pub extern "C" fn print_rust(message: *const libc::c_char) {
    // Keep behavior consistent with the binary: print via logger when configured.
    // If no logger is configured, this becomes a no-op.
    if message.is_null() {
        return;
    }
    let message = unsafe { std::ffi::CStr::from_ptr(message) };
    if let Ok(message) = message.to_str() {
        log::info!("{}", message);
    }
}

fn compress_to_gz(data: Vec<u8>) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&data).unwrap();
    encoder.finish().unwrap()
}

fn decompress_from_gz(compressed_data: Vec<u8>) -> Vec<u8> {
    let mut decoder = GzDecoder::new(compressed_data.as_slice());
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data).unwrap();
    decompressed_data
}

pub fn save<T: serde::Serialize>(obj: &T, path: &str) {
    let data = bincode::serialize(obj).unwrap();
    std::fs::write(path, compress_to_gz(data)).unwrap();
}

pub fn read_gz(path: &str) -> Vec<u8> {
    decompress_from_gz(std::fs::read(path).unwrap())
}

pub fn read<'de, T: serde::Deserialize<'de>>(data: &'de [u8]) -> T {
    bincode::deserialize(data).unwrap()
}

pub fn remove_old_states(program_args: &ProgramArgs, keep_states: i32, max_frame: i32) {
    if keep_states > 0 {
        let mut n = 0;
        for i in 0..=max_frame {
            let path = format!("{}/state_{}.bin.gz", program_args.output, i);
            if std::path::Path::new(&path).exists() {
                n += 1;
            }
        }
        log::info!("Removing old states...");
        let mut i = 0;
        while n > keep_states {
            let path = format!("{}/state_{}.bin.gz", program_args.output, i);
            if std::path::Path::new(&path).exists() {
                log::info!("Removing {}...", path);
                std::fs::remove_file(path).unwrap_or(());
                n -= 1;
            }
            i += 1;
            if i > max_frame {
                break;
            }
        }
    }
}

pub fn remove_old_files(
    dirpath: &str,
    prefix: &str,
    suffix: &str,
    keep_number: i32,
    max_frame: i32,
) {
    if keep_number <= 0 {
        return;
    }

    let keep_number = keep_number.max(1);

    let mut n = 0;
    for i in 0..=max_frame {
        let path = format!("{}/{}{}{}", dirpath, prefix, i, suffix);
        if std::path::Path::new(&path).exists() {
            n += 1;
        }
    }

    let mut i = 0;
    while n > keep_number {
        let filename = format!("{}{}{}", prefix, i, suffix);
        let path = format!("{}/{}", dirpath, filename);
        if std::path::Path::new(&path).exists() {
            log::info!("Removing {}...", filename);
            std::fs::remove_file(path).unwrap_or(());
            n -= 1;
        }
        i += 1;
        if i > max_frame {
            break;
        }
    }
}
