// Simple AI Training & Inference Demo using Burn with GPU (wgpu backend)
// Trains a neural network to learn the XOR function

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    module::{AutodiffModule, Module},
    nn::{self, loss::MseLoss, Linear, LinearConfig, Relu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, ElementConversion, Tensor, TensorData},
};

// Type aliases for cleaner code
type Backend = Wgpu;
type AutodiffBack = Autodiff<Backend>;

// Simple MLP (Multi-Layer Perceptron) for XOR learning
#[derive(Module, Debug)]
struct MlpModel<B: burn::tensor::backend::Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    activation: Relu,
}

impl<B: burn::tensor::backend::Backend> MlpModel<B> {
    fn new(device: &B::Device) -> Self {
        // XOR: 2 inputs -> 8 hidden -> 8 hidden -> 1 output
        let fc1 = LinearConfig::new(2, 8).with_bias(true).init(device);
        let fc2 = LinearConfig::new(8, 8).with_bias(true).init(device);
        let fc3 = LinearConfig::new(8, 1).with_bias(true).init(device);
        let activation = Relu::new();

        Self { fc1, fc2, fc3, activation }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.fc1.forward(x));
        let x = self.activation.forward(self.fc2.forward(x));
        self.fc3.forward(x)
    }
}

// Training step
fn train_step<B: AutodiffBackend>(
    model: MlpModel<B>,
    optimizer: &mut impl Optimizer<MlpModel<B>, B>,
    inputs: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> (MlpModel<B>, f32) {
    // Forward pass
    let predictions = model.forward(inputs);

    // Compute MSE loss
    let loss = MseLoss::new().forward(predictions, targets.clone(), nn::loss::Reduction::Mean);
    let loss_value = loss.clone().into_scalar().elem::<f32>();

    // Backward pass
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);

    // Update weights
    let model = optimizer.step(0.01, model, grads);

    (model, loss_value)
}

fn main() {
    println!("===========================================");
    println!("  Burn AI Demo - XOR Learning with GPU");
    println!("===========================================\n");

    // Initialize GPU device
    let device = WgpuDevice::default();
    println!("Using device: {:?}\n", device);

    // Create XOR training data
    // XOR truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    let inputs_data: Vec<f32> = vec![
        0.0, 0.0,  // -> 0
        0.0, 1.0,  // -> 1
        1.0, 0.0,  // -> 1
        1.0, 1.0,  // -> 0
    ];
    let targets_data: Vec<f32> = vec![0.0, 1.0, 1.0, 0.0];

    // Create model and optimizer
    let model: MlpModel<AutodiffBack> = MlpModel::new(&device);
    let mut optimizer = AdamConfig::new().init();

    println!("Model architecture:");
    println!("  Input:  2 neurons (XOR inputs)");
    println!("  Hidden: 8 neurons (ReLU)");
    println!("  Hidden: 8 neurons (ReLU)");
    println!("  Output: 1 neuron\n");

    println!("Training on XOR function...\n");

    // Training loop
    let epochs = 1000;
    let mut model = model;

    for epoch in 0..epochs {
        // Create tensors for this epoch with explicit shape
        let inputs: Tensor<AutodiffBack, 2> =
            Tensor::from_data(TensorData::new(inputs_data.clone(), [4, 2]), &device);
        let targets: Tensor<AutodiffBack, 2> =
            Tensor::from_data(TensorData::new(targets_data.clone(), [4, 1]), &device);

        let (updated_model, loss) = train_step(model, &mut optimizer, inputs, targets);
        model = updated_model;

        // Print progress every 100 epochs
        if epoch % 100 == 0 || epoch == epochs - 1 {
            println!("  Epoch {:4} | Loss: {:.6}", epoch, loss);
        }
    }

    println!("\n===========================================");
    println!("  Inference Results");
    println!("===========================================\n");

    // Switch to inference mode (no autodiff needed)
    let model_valid = model.valid();

    // Test each XOR combination
    let test_cases = vec![
        ([0.0_f32, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    println!("  Input     | Expected | Predicted | Rounded");
    println!("  ----------|----------|-----------|--------");

    for (input, expected) in test_cases {
        let input_tensor: Tensor<Backend, 2> =
            Tensor::from_data(TensorData::new(input.to_vec(), [1, 2]), &device);

        let output = model_valid.forward(input_tensor);
        let predicted: f32 = output.into_scalar().elem();
        let rounded = if predicted > 0.5 { 1.0 } else { 0.0 };
        let correct = if rounded == expected { "OK" } else { "X" };

        println!(
            "  [{}, {}]   |    {}     |   {:.4}   |   {}  {}",
            input[0] as i32, input[1] as i32, expected as i32, predicted, rounded as i32, correct
        );
    }

    println!("\n===========================================");
    println!("  Training complete!");
    println!("===========================================");
}
