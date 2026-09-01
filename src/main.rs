use std::{f64::consts::PI, vec};

use anyhow::Result;
use candle_core::{Device, Tensor};

trait TensorExt {
    fn z_norm(&self) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn z_norm(&self) -> Result<Tensor> {
        let data = self.clone();

        let mean = self.mean(0)?; // (1, dims)
        let diff = data.broadcast_sub(&mean)?; // (instance, dims)
        let variance = diff.sqr()?.mean(0)?; // (1, dims)
        let stdev = (variance + 1e-8)?.sqrt()?;

        println!("----- debugging norm -----");
        println!("mean: {}\n\ndiff: {}\n\nstd: {}", mean, diff, stdev);

        let norm = diff.broadcast_div(&stdev)?;

        println!("norm_data is {}", norm);
        println!("----- end debugging norm -----");
        Ok(norm)
    }
}

fn load_data(device: &Device) -> Result<Tensor> {
    let data: &[f64] = &[
        1.1, 1.3, 1.5, 1.6, 1.1, 1.3, 1.5, 1.6, 1.2, 7.3, 1.2, 2.3, 1.1, 3.3, 1.5, 1.6, 1.2, 1.3,
        1.2, 2.3, 1.2, 1.3, 1.2, 2.3,
    ];
    let tensor = Tensor::from_slice(data, (6, 4), device)?;
    Ok(tensor)
}

fn p_x(
    x: &Tensor,
    mean: &Tensor,
    two_variance: &Tensor,
    two_pi_sqrt_std_dev: &Tensor,
) -> Result<f64> {
    let px = x
        .broadcast_sub(mean)?
        .sqr()?
        .broadcast_div(two_variance)?
        .exp()?
        .broadcast_mul(two_pi_sqrt_std_dev)?
        .recip()?;
    let pax: Vec<f64> = px.to_vec1::<f64>()?;
    let px = pax.into_iter().fold(1.0, |accumulate, x| accumulate * x);
    Ok(px)
}

fn main() -> Result<()> {
    println!("Hello, world!");
    println!("\n[CONFIG]");
    let device: Device = Device::Cpu;
    let data: Tensor = load_data(&device)?;
    let threadshold = 1e-3;
    println!("device: {:?},\nthreadshold: {threadshold}", device);
    println!("[END CONFIG]\n");
    println!("Before normalize");
    println!("{}", data);
    // let _ = data.printSelf()?;
    let data = data.z_norm()?;
    let variance_f64: Tensor = Tensor::new(1.0, &device)?;
    let stdev_f64: Tensor = Tensor::new(1.0, &device)?;
    let mean: Tensor = Tensor::new(0.0, &device)?;

    println!("After normalize");
    // println!("{data},\nmean: {mean},\nstdev: {stdev}\nvariance: {variance}\n");

    let two_variance: Tensor = variance_f64.broadcast_mul(&Tensor::new(2.0, &device)?)?;
    let two_pi_sqrt_std_dev: Tensor =
        stdev_f64.broadcast_mul(&Tensor::new(2.0 * PI, &device)?.sqrt()?)?;

    println!(
        "two_variance: {} two_pi_sqrt_std_dev: {:?} ",
        two_variance, two_pi_sqrt_std_dev
    );
    let data_shape = data.shape();
    let dims2 = data_shape.dims2()?;
    println!(
        "data: {} data_shape: {:?} data_shape_dims2: {:?}",
        data, data_shape, dims2
    );

    let rows = dims2.0;

    let mut anomalies: Vec<f64> = vec![];

    // println!(
    //     "row: {:?}, column {:?}, vec: {:?}",
    //     rows, columns, anomalies
    // );

    for row in 0..rows {
        let row_tensor = data
            .index_select(&Tensor::new(&[row as u32], &device)?, 0)?
            .squeeze(0)?;
        let px = p_x(&row_tensor, &mean, &two_variance, &two_pi_sqrt_std_dev)?;
        //for now just push everything;

        if px < threadshold {
            println!("Anomaly at {}", row + 1);
            anomalies.push(px * 100f64);
        }
    }

    println!("p_x: {:?}", anomalies);
    Ok(())
}
