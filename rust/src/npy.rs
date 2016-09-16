use regex::{Regex, Captures};
use ieee754::Ieee754;
use zip::ZipArchive;

use std::error::Error;
use std::io;
use std::io::Read;
use std::fs::File;
use std::path::Path;
use std::str;

#[derive(Clone)]
/// The data table we read from the .npy files.
pub struct Table {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

/// Read a .npz file, and return the table for each file.
pub fn read_npz_file(path: &Path) -> Result<Vec<Table>, io::Error> {
    let file = try!(File::open(path));
    let mut archive = try!(ZipArchive::new(file));

    let mut tables: Vec<_> = (0..archive.len())
        .flat_map(|i| {
            let file = archive.by_index(i).unwrap();
            let filename = file.name().to_string();
            read_npy_file(file).map(|table| {
                let num = extract_part_number(&filename);
                (filename, num.unwrap_or(0), table)
            })
        })
        .collect();
    tables.sort_by_key(|t| t.1);

    Ok(tables.into_iter()
        .map(|(_, _, table)| table)
        .collect())
}

fn read_npy_file<R>(mut file: R) -> Result<Table, String>
    where R: Read
{
    let mut bytes = Vec::new();
    try!(file.read_to_end(&mut bytes).map_err(|e| e.description().to_string()));
    parse_npy_file(&bytes)
}

fn parse_npy_file(bytes: &Vec<u8>) -> Result<Table, String> {
    const START: &'static [u8] = b"\x93NUMPY";
    for i in 0..6 {
        if bytes[i] != START[i] {
            let e = format!("File did not start with \'\\x93NUMPY\': {:?}", &bytes[..6]);
            return Err(e.to_string());
        }
    }
    let major_version = bytes[6];
    let minor_version = bytes[7];
    if (major_version, minor_version) != (1, 0) {
        return Err("Only support .npy files of version 1.0".to_string());
    }
    let header_len = bytes_to_u16_le(&bytes[8..]);
    let header = &bytes[10..10 + header_len as usize];
    let header_str = str::from_utf8(header).unwrap();

    // We will assume "fortran_order" is false :)
    lazy_static! {
        static ref RE: Regex = Regex::new(".*\'descr\': ?'<f(\\d).*\'shape\': ?\\((\\d*), *(\\d*)\\)").unwrap();
    }
    let capture_to_n = |c: &Captures, n| c.at(n).and_then(|s| s.parse().ok()).unwrap_or(1);
    let (float_precision, height, width) = RE.captures_iter(header_str)
        .next()
        .map(|capt| (capture_to_n(&capt, 1), capture_to_n(&capt, 2), capture_to_n(&capt, 3)))
        .unwrap_or((0, 0, 0));

    let mut vec: Vec<f64> = Vec::with_capacity(width * height);
    let data = &bytes[10 + header_len as usize..];
    let mut i = 0;

    match float_precision {
        4 => {
            for _ in 0..width * height {
                let f = bytes_to_f32_le(&data[i..]);
                vec.push(f as f64);
                i += 4;
            }
        }
        8 => {
            for _ in 0..width * height {
                let f = bytes_to_f64_le(&data[i..]);
                vec.push(f);
                i += 8;
            }
        }
        _ => panic!(format!("float_precision: {}", float_precision)),
    }

    Ok(Table {
        data: vec,
        width: width,
        height: height,
    })
}

fn bytes_to_u16_le(bytes: &[u8]) -> u16 {
    bytes[0] as u16 | (bytes[1] as u16) << 8
}

fn bytes_to_f32_le(bytes: &[u8]) -> f32 {
    let num = (bytes[3] as u32) << 24 | (bytes[2] as u32) << 16 | (bytes[1] as u32) << 8 |
              bytes[0] as u32;

    let si_mask = 0b10000000000000000000000000000000;
    let ex_mask = 0b01111111100000000000000000000000;
    let sg_mask = 0b00000000011111111111111111111111;
    let sign = num & si_mask == si_mask;
    let exponent = ((num & ex_mask) >> 23) as i16 - 127;
    let significand = num & sg_mask;
    f32::recompose(sign, exponent, significand)
}
fn bytes_to_f64_le(bytes: &[u8]) -> f64 {
    let num = (bytes[7] as u64) << 56 | (bytes[6] as u64) << 48 | (bytes[5] as u64) << 40 |
              (bytes[4] as u64) << 32 | (bytes[3] as u64) << 24 |
              (bytes[2] as u64) << 16 | (bytes[1] as u64) << 8 | bytes[0] as u64;

    let si_mask: u64 = 0b1000000000000000000000000000000000000000000000000000000000000000;
    let ex_mask: u64 = 0b0111111111110000000000000000000000000000000000000000000000000000;
    let sg_mask: u64 = 0b0000000000001111111111111111111111111111111111111111111111111111;
    let sign = num & si_mask == si_mask;
    let exponent = ((num & ex_mask) >> 52) as i16 - 1023;
    let significand = num & sg_mask;
    f64::recompose(sign, exponent, significand)
}

fn extract_part_number(s: &str) -> Option<usize> {
    lazy_static! {
        static ref RE: Regex = Regex::new("[^0-9]*([0-9]+).*").unwrap();
    }
    RE.captures(s).and_then(|caps| caps.at(1)).and_then(|num| num.parse().ok())
}
