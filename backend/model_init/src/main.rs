use std::env;
use std::path::Path;

use std::io;

use anyhow::{Result, bail};
use tch::{CModule, Device, Tensor, Kind, Cuda};

enum ModelFormat {
    TorchScript,
    Export,
}

fn detect_format(path: &str) -> Result<ModelFormat> {
    let mut input_text = String::new();
    println!("Please enter in what type of model you'd like to initialize\n1. torchscript\n2. export\n3. infer");
    io::stdin()
    .read_line(&mut input_text)
    .expect("Failed to read line");
    let input_text = input_text.trim();
    if (input_text == "1") || (input_text == "3" && path.ends_with(".pt")) {
        Ok(ModelFormat::TorchScript)
    } else if (input_text == "2") || (input_text == "3" && (path.ends_with(".pte") || path.ends_with(".pt2"))) {
        Ok(ModelFormat::Export)
    } else {
        bail!("Unsupported model format")
    }
}

fn run_torchscript(path: &str, device: Device) -> Result<()> {
    println!("Loading TorchScript model: {}", path);
    let model = CModule::load_on_device(path, device)?;
    println!("Model loaded.");
    let input = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, device));
    println!("Running inference...");
    let output = model.forward_ts(&[input])?;
    let prediction = output.argmax(-1, false).int64_value(&[0]);
    println!("Predicted digit: {}", prediction);
    Ok(())
}

fn run_export(_path: &str) -> Result<()> {
    println!("Detected .pte model");
    println!("ExecuTorch loading not implemented yet.");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run -- <model_file>");
        return Ok(());
    }
    let model_path = &args[1];
    if !Path::new(model_path).exists() {
        bail!("Model file not found");
    }
    println!("CUDA available: {}", Cuda::is_available());
    println!("CUDA devices: {}", Cuda::device_count());
    let device = if Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);
    match detect_format(model_path)? {
        ModelFormat::TorchScript => {
            run_torchscript(model_path, device)?;
        }
        ModelFormat::Export => {
            run_export(model_path)?;
        }
    }
    Ok(())
}