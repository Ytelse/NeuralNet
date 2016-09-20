#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate itertools;
extern crate regex;
extern crate ieee754;
extern crate zip;
extern crate rayon;
macro_rules! print_var {
    ($v: ident) => {{
        println!("{} = {}", stringify!($v), $v);
    }}
}

mod npy;
mod mnist;

use rayon::prelude::*;
use std::io;
use std::path::Path;

type Layer = Vec<Neuron>;
#[derive(Debug)]
struct Neuron {
    in_weights: Vec<f64>,
    bias: f64,
    beta: f64,
    gamma: f64,
    mean: f64,
    inv_stddev: f64,
    f: f64,
}

impl Neuron {
    fn process(&self, input_values: &[f64]) -> f64 {
        assert!(input_values.len() == self.in_weights.len());
        let x: f64 = input_values.iter()
            .zip(&self.in_weights)
            .map(|(x, w)| x * w)
            .sum();
        // let batch_norm = self.gamma * self.inv_stddev * (input - self.mean) + self.beta;
        // batch_norm.signum()
        let a = self.gamma * self.inv_stddev;
        let b = self.mean - self.beta / a;
        ((x - b) * a.signum()).signum();
        (x + self.f).signum()
    }
}


#[derive(Debug)]
struct NeuralNetwork {
    num_layers: usize,
    neuron_count_in_layers: Vec<usize>,
    input_count: usize,
    output_count: usize,
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn read_from_npz(path: &Path) -> Result<Self, io::Error> {
        let items = try!(npy::read_npz_file(path));

        let layers: Vec<Layer> = items.chunks(6)
            .map(|chunk| {
                let weights = &chunk[0];
                let biases = &chunk[1];
                let betas = &chunk[2];
                let gammas = &chunk[3];
                let means = &chunk[4];
                let inv_stddevs = &chunk[5];

                let weights_data = vec_transpose_n(&weights.data, weights.width);

                izip!(weights_data.chunks(weights.height),
                      &biases.data,
                      &betas.data,
                      &gammas.data,
                      &means.data,
                      &inv_stddevs.data)
                    .map(|(w, bi, be, ga, m, i)| {
                        let mut weights = Vec::with_capacity(w.len());
                        weights.extend(w);
                        weights.iter_mut().map(|n: &mut f64| *n = n.signum()).last();
                        let a = *ga * *i;
                        let b = *be - a * *m;
                        let mut f = (b / a).floor();
                        if f > 127.0 {
                            f = 127.0;
                        }
                        if f < -128.0 {
                            f = -128.0
                        }
                        // println!("{}", f);
                        Neuron {
                            in_weights: weights,
                            bias: *bi,
                            beta: *be,
                            gamma: *ga,
                            mean: *m,
                            inv_stddev: *i,
                            f: f,
                        }
                    })
                    .collect()
            })
            .collect();
        let neuron_count_in_layers: Vec<_> = layers.iter().map(|layer| layer.len()).collect();

        Ok(NeuralNetwork {
            num_layers: layers.len(),
            input_count: layers[0][0].in_weights.len(),
            output_count: neuron_count_in_layers.last().cloned().unwrap_or(0),
            neuron_count_in_layers: neuron_count_in_layers,
            layers: layers,
        })
    }

    fn process_input(&self, input: &Vec<f64>) -> Option<u8> {
        if input.len() != self.input_count {
            panic!("input.len() != self.input_count: {} != {}",
                   input.len(),
                   self.input_count);
        }

        self.layers
            .iter()
            .fold(input.clone(), |input_vector, layer| {
                let mut v = Vec::with_capacity(layer.len());
                layer.par_iter()
                    .map(|n| n.process(&input_vector))
                    .collect_into(&mut v);
                v
            })
            .iter()
            .position(|&n| n > 0.0)
            .map(|n| n as u8)
    }
}


fn vec_transpose_n<T>(vec: &Vec<T>, width: usize) -> Vec<T>
    where T: Clone
{
    let mut v = Vec::with_capacity(vec.len());
    let mut start = 0;
    let mut i = 0;
    while start < width {
        v.push(vec[i].clone());
        i += width;
        if i >= vec.len() {
            start += 1;
            i = start;
        }
    }
    v
}

fn main() {
    let image_path = Path::new("../image_and_label_sets/train-images.idx3-ubyte");
    let labels_path = Path::new("../image_and_label_sets/train-labels.idx1-ubyte");
    let network_path_large = Path::new("../networks/1024-128.npz");
    let network_path_small = Path::new("../networks/256.npz");
    if let (Ok(images), Ok(network)) = (mnist::read_image_label_pair(image_path, labels_path),
                                        NeuralNetwork::read_from_npz(network_path_large)) {
        let n_images = 60000;
        let successes = images.iter()
            .take(n_images)
            .filter(|img| {
                let pixels: Vec<f64> = img.float_data()
                    .into_iter()
                    .map(|n| if n > 127.0 {
                        1.
                    } else {
                        -1.
                    })
                    .collect();
                network.process_input(&pixels)
                    .map(|label| label == img.label)
                    .unwrap_or(false)
            })
            .count();
        println!("Was right {}/{} times ({}%)",
                 successes,
                 n_images,
                 100.0 * successes as f64 / n_images as f64);
    } else {
        println!("There is something wrong with your paths :)");
    }
}
