use anyhow::Result;
use candle_core::{Device, Tensor};
// use std::vec;

trait TensorExt {
    fn printSelf(&self) -> Result<()>;
    fn z_norm(&self) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn printSelf(&self) -> Result<()> {
        let num_rows = self.dims()[1];
        let num_cols = self.dims()[2];
        for i in 0..num_rows {
            println!("\n-------");
            for j in 0..num_cols {
                let element = self.get(0)?.get(i)?.get(j)?;
                print!("\nElement is {}", element);
            }
        }
        Ok(())
    }

    fn z_norm(&self) -> Result<Tensor> {
        let data = self.clone();

        let mean = self.mean_all()?;
        let diff = data.broadcast_sub(&mean)?;
        let variance = data.flatten_all()?.var(0)?;
        let stdev = (variance + 1e-8)?.sqrt()?;

        // println!("----- debugging norm -----");
        // println!("mean: {}\n\ndiff: {}\n\nstd: {}", mean, diff, stdev);

        let norm = diff.broadcast_div(&stdev)?;

        // println!("----- debugging norm -----");
        // println!("norm_data is {}", norm);
        Ok(norm)
    }
}

fn load_data(device: &Device) -> Result<Tensor> {
    let data: &[f32] = &[1.1, 1.3, 1.5, 1.6, 1.2, 7.3, 1.2, 2.3];
    let tensor = Tensor::from_slice(data, (1, 2, 4), device)?;
    Ok(tensor)
}

fn main() -> Result<()> {
    println!("Hello, world!");
    let device: Device = Device::Cpu;
    let data: Tensor = load_data(&device)?;
    println!("Before normalize");
    println!("{}", data);
    // let _ = data.printSelf()?;
    let norm_data: Tensor = data.z_norm()?;
    println!("After normalize");
    println!("{}", norm_data);
    Ok(())
}
