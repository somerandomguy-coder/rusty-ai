use std::f64::consts::PI;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
// use std::vec;

trait TensorExt {
    #[warn(dead_code)]
    fn print_self(&self) -> Result<()>;

    fn z_norm(&self) -> Result<(Tensor, Tensor, Tensor, Tensor)>;
}

impl TensorExt for Tensor {
    fn print_self(&self) -> Result<()> {
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

    fn z_norm(&self) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let data = self.clone();

        let mean = self.mean_all()?;
        let diff = data.broadcast_sub(&mean)?;
        let variance = data.flatten_all()?.var(0)?;
        let variance_copy = variance.copy()?;
        let stdev = (variance + 1e-8)?.sqrt()?;

        // println!("----- debugging norm -----");
        // println!("mean: {}\n\ndiff: {}\n\nstd: {}", mean, diff, stdev);

        let norm = diff.broadcast_div(&stdev)?;

        // println!("----- debugging norm -----");
        // println!("norm_data is {}", norm);
        Ok((norm, mean, stdev, variance_copy))
    }
}

fn load_data(device: &Device) -> Result<Tensor> {
    let data: &[f32] = &[
        1.1, 1.3, 1.5, 1.6, 1.2, 7.3, 1.2, 2.3, 1.1, 7.3, 1.5, 1.6, 1.2, 1.3, 1.2, 2.3,
    ];
    let tensor = Tensor::from_slice(data, (4, 4), device)?;
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
    let device: Device = Device::Cpu;
    let data: Tensor = load_data(&device)?;
    println!("Before normalize");
    println!("{}", data);
    // let _ = data.printSelf()?;
    let tuple_norm = data.z_norm()?;
    let norm_data = tuple_norm.0;
    let mean = tuple_norm.1;
    let stdev = tuple_norm.2;
    let variance = tuple_norm.3;
    let variance_f64: Tensor = variance.to_dtype(DType::F64)?;
    let stdev_f64: Tensor = stdev.to_dtype(DType::F64)?;

    println!("After normalize");
    println!("{norm_data},\nmean: {mean},\nstdev: {stdev}\nvariance: {variance}\n");

    let two_variance: Tensor = variance_f64.broadcast_mul(&Tensor::new(2.0, &device)?)?;
    let two_pi_sqrt_std_dev: Tensor =
        stdev_f64.broadcast_mul(&Tensor::new(2.0 * PI, &device)?.sqrt()?)?;

    let data_shape = data.shape();
    let dims2 = data_shape.dims2()?;
    println!(
        "data: {} data_shape: {:?} data_shape_dims2: {:?}",
        data, data_shape, dims2
    );

    let rows = data.shape().dims2()?.0;
    // for row in 0..rows {

    // }

    Ok(())
}
