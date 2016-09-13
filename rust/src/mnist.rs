use std::path::Path;
use std::fs::File;
use std::io::{Error, Read};
use std::iter::repeat;

fn read_u32(file: &mut File) -> u32 {
    let mut buff = [0; 4];
    let _ = file.read_exact(&mut buff);
    (buff[0] as u32) << 24 | (buff[1] as u32) << 16 | (buff[2] as u32) << 8 | (buff[3] as u32)
}

pub struct Image {
    pub width: u32,
    height: u32,
    data: Vec<u8>,
    pub label: u8,
}

impl Image {
    pub fn float_data(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&u| if u > 127 {
                255.0
            } else {
                0.0
            })
            .collect()
    }

    pub fn print(&self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let i = (y * self.height + x) as usize;
                let c = match (255 - self.data[i]) / 29 {
                    0 => '@',
                    1 => '#',
                    2 => '8',
                    3 => '&',
                    4 => 'o',
                    5 => ':',
                    6 => '*',
                    7 => '.',
                    8 => ' ',
                    _ => unreachable!(),
                };
                print!("{} ", c);
            }
            print!("\n");
        }
    }
}

fn read_label_file(path: &Path) -> Result<Vec<u8>, Error> {
    let mut file = try!(File::open(path));
    let magic_number = read_u32(&mut file);
    assert!(magic_number == 0x0801);
    let n_labels = read_u32(&mut file) as u64;
    let mut buffer = Vec::with_capacity(n_labels as usize);
    file.take(n_labels)
        .read_to_end(&mut buffer)
        .map(|_| buffer)
}

fn read_image_file(path: &Path) -> Result<Vec<Image>, Error> {
    let mut file = try!(File::open(path));
    let magic_number = read_u32(&mut file);
    assert!(magic_number == 0x0803);
    let n_images = read_u32(&mut file);
    let image_height = read_u32(&mut file);
    assert!(image_height == 28);
    let image_width = read_u32(&mut file);
    assert!(image_width == 28);

    let images = (0..n_images)
        .map(|_| {
            let size = image_height * image_width;
            let mut vec = repeat(0).take(size as usize).collect::<Vec<u8>>();
            let _ = file.read_exact(&mut vec.as_mut_slice());
            assert!(vec.len() == size as usize);
            Image {
                width: image_width as u32,
                height: image_height as u32,
                data: vec,
                label: 255,
            }

        })
        .collect();
    Ok(images)
}

pub fn read_image_label_pair(img_path: &Path, lab_path: &Path) -> Result<Vec<Image>, Error> {
    let mut images = try!(read_image_file(img_path));
    let labels = try!(read_label_file(lab_path));
    for (mut img, label) in images.iter_mut().zip(labels) {
        img.label = label;
    }
    Ok(images)
}
