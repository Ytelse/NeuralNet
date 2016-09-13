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
    pub data: Vec<f32>,
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

    // TODO: Fix this:
    // We'll take a shortcut, and simply search for "shape" in the header,
    // in order to read out the array dimensions.
    // We will assume "dtype" is f4, and "fortran_order" is false :)
    lazy_static! {
        static ref RE: Regex = Regex::new(".*\'shape\': ?\\((\\d*), *(\\d*)\\)").unwrap();
    }
    let capture_to_n = |c: &Captures, n| c.at(n).and_then(|s| s.parse().ok()).unwrap_or(1);
    let (height, width) = RE.captures_iter(header_str)
        .next()
        .map(|capt| (capture_to_n(&capt, 1), capture_to_n(&capt, 2)))
        .unwrap_or((0, 0));

    let mut vec: Vec<f32> = Vec::with_capacity(width * height);
    let data = &bytes[10 + header_len as usize..];
    let mut i = 0;
    for _ in 0..width * height {
        let f = bytes_to_f32_le(&data[i..]);
        vec.push(f);
        i += 4;
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

fn extract_part_number(s: &str) -> Option<usize> {
    lazy_static! {
        static ref RE: Regex = Regex::new("[^0-9]*([0-9]+).*").unwrap();
    }
    RE.captures(s).and_then(|caps| caps.at(1)).and_then(|num| num.parse().ok())
}
