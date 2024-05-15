use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor, Device, nn::VarStore, Kind, COptimizer};
use mnist::*;
use tqdm::tqdm;
use std::time::Instant;

// load mnist dataset and preprocess it
fn load_mnist_data() -> (Tensor, Tensor, Tensor, Tensor) {
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_images_tensor: Tensor = Tensor::from_slice(&trn_img.into_iter()
        .map(|x| x as f32 / 255.0)
        .collect::<Vec<f32>>())
        .view([-1, 28, 28]);

    let train_labels_tensor: Tensor = Tensor::from_slice(&trn_lbl.into_iter()
        .map(|x| x as i64)
        .collect::<Vec<i64>>());

    let test_images_tensor: Tensor = Tensor::from_slice(&tst_img.into_iter()
        .map(|x| x as f32 / 255.0)
        .collect::<Vec<f32>>())
        .view([-1, 28, 28]);

    let test_labels_tensor: Tensor = Tensor::from_slice(&tst_lbl.into_iter()
        .map(|x| x as i64)
        .collect::<Vec<i64>>());

    (train_images_tensor, train_labels_tensor, test_images_tensor, test_labels_tensor)
}

// Function to create the neural network model.
fn create_network(vs: &nn::Path) -> nn::Sequential {
    let input_size: i64 = 784;
    let hidden_size: i64 = 128;
    let output_size: i64 = 10;
    let cfg: nn::LinearConfig = nn::LinearConfig::default();

    nn::seq()
        .add(nn::linear(vs / "layer1", input_size, hidden_size, cfg))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer2", hidden_size, output_size, cfg))
}

// function to run the training loop.
fn train_network(net: &mut nn::Sequential,
                 optimizer: &mut nn::Optimizer,
                 train_images_tensor: &Tensor,
                 train_labels_tensor: &Tensor,
                 num_epochs: i32,
                 batch_size: i64) {
    let num_batches = train_images_tensor.size()[0] / batch_size;
    let mut epoch_loss = 0.0;
    for epoch in tqdm(1..=num_epochs) {
        let mut total_correct = 0;
        let mut total = 0;
        for batch_index in tqdm(0..num_batches) {
            let batch = train_images_tensor.narrow(0, batch_index * batch_size, batch_size).view([-1, 28 * 28]);
            let labels = train_labels_tensor.narrow(0, batch_index * batch_size, batch_size);
            let predictions = net.forward(&batch);
            let loss = predictions.cross_entropy_for_logits(&labels);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_correct += label_accuracy(&predictions, &labels) as i64;
            total += batch_size;
            epoch_loss = loss.double_value(&[]);
        }
        println!("Epoch: {}, Loss: {}", epoch, epoch_loss);
        println!("Train accuracy: {}", total_correct as f64 / total as f64);
    }
}

fn evaluate_network(net: &mut nn::Sequential,
                    test_images_tensor: &Tensor,
                    test_labels_tensor: &Tensor,
                    test_batch_size: i64) -> f64 {
    let num_test_batches = test_images_tensor.size()[0] / test_batch_size;
    let mut total_correct = 0;
    let mut total = 0;

    for batch_index in tqdm(0..num_test_batches) {
        let batch = test_images_tensor.narrow(0, batch_index * test_batch_size, test_batch_size).view([-1, 28 * 28]);
        let labels = test_labels_tensor.narrow(0, batch_index * test_batch_size, test_batch_size);
        let logits = net.forward(&batch);

        total_correct += label_accuracy(&logits, &labels) as i64;
        total += test_batch_size;
    }

    total_correct as f64 / total as f64
}

fn label_accuracy(predictions: &Tensor, labels: &Tensor) -> i64 {
    let predicted_labels = predictions.argmax(-1, false);
    let correct_predictions = predicted_labels.eq_tensor(&labels);

    correct_predictions.to_kind(Kind::Int64).sum(Kind::Int64).int64_value(&[])
}

fn main() {
    let device = Device::Cpu;
    let vs = VarStore::new(device);
    let (train_images_tensor, train_labels_tensor, test_images_tensor, test_labels_tensor) = load_mnist_data();
    let mut net = create_network(&vs.root());
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let num_epochs = 10;
    let batch_size = 64;

    let start_time = Instant::now();
    train_network(&mut net, &mut optimizer, &train_images_tensor, &train_labels_tensor, num_epochs, batch_size);
    let elapsed = start_time.elapsed();
    println!("Training time: {:?}", elapsed);
    let test_accuracy = evaluate_network(&mut net, &test_images_tensor, &test_labels_tensor, batch_size);
    println!("Test accuracy: {}", test_accuracy);
}
