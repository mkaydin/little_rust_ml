use std::error::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use tch::{nn, Device, Kind, Tensor, nn::Module, nn::OptimizerConfig};
use std::time::Instant;
use image::{DynamicImage, imageops::FilterType};

#[derive(Debug)]
struct ImageDataset {
    images: Vec<PathBuf>,
    labels: Vec<String>,
    class_names: Vec<String>,
}

impl ImageDataset {
    fn new(data_dir: &str, classes: &[&str]) -> Self {
        let mut images = Vec::new();
        let mut labels = Vec::new();
        let mut class_names = Vec::new();

        for &class_name in classes {
            class_names.push(class_name.to_string());
            let class_dir = Path::new(data_dir).join(class_name);
            for entry in WalkDir::new(&class_dir).into_iter().filter_map(Result::ok) {
                if entry.path().is_file() {
                    images.push(entry.path().to_path_buf());
                    labels.push(class_name.to_string());
                }
            }
        }

        ImageDataset {
            images,
            labels,
            class_names,
        }
    }

    fn len(&self) -> usize {
        self.images.len()
    }
}

fn data_loader(dataset: &ImageDataset, batch_size: usize, shuffle: bool) -> Vec<(Tensor, Tensor)> {
    let mut indices: Vec<usize> = (0..dataset.len()).collect();
    if shuffle {
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
    }

    indices.chunks(batch_size)
        .filter_map(|batch_indices| {
            let images: Result<Vec<Tensor>, _> = batch_indices.iter().map(|&i| {
                let img = image::open(&dataset.images[i])?; // Open the image file
                let img = img.resize(224, 224, FilterType::Nearest); // Resize and specify filter type
                load_image(&img) // Pass the DynamicImage to load_image
            }).collect();
            match images {
                Ok(images) => {
                    let labels: Vec<i64> = batch_indices.iter().map(|&i| dataset.class_names.iter().position(|r| r == &dataset.labels[i]).unwrap() as i64).collect();
                    Some((Tensor::cat(&images, 0), Tensor::from_slice(&labels)))
                },
                Err(_) => None
            }
        })
        .collect()
}
/*
// Corrected load_image function
fn load_image(image: &DynamicImage) -> Result<Tensor, Box<dyn Error>> {
    // Ensure the image is of size 224x224
    let resized_image = image.resize_exact(224, 224, image::imageops::FilterType::Nearest);

    // Convert the DynamicImage to a Vec<u8> in RGB format
    let rgb_pixels: Vec<u8> = resized_image.to_rgb8().into_raw();

    // Convert the Vec<u8> to a Vec<f32> and normalize to [0.0, 1.0]
    let img_floats: Vec<f32> = rgb_pixels.iter().map(|&pixel| pixel as f32 / 255.0).collect();

    // Check if the img_floats has the correct number of elements
    if img_floats.len() != 224 * 224 * 3 {
        return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Image has unexpected number of elements")));
    }

    // Use f_from_slice to create a tensor from the float slice
    let img_tensor = Tensor::f_from_slice(&img_floats)?;

    // Reshape the tensor to the expected input shape for your network
    let img_tensor = img_tensor.view([1, 3, 224, 224]);

    // Check if the image has the expected dimensions
    if img_tensor.size() != &[1, 3, 224, 224] {
        return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Image has unexpected dimensions")));
    }

    Ok(img_tensor)
}

fn create_network(vs: &nn::Path) -> nn::Sequential {
    let net = nn::seq()
        .add(nn::conv2d(vs / "conv1", 3, 16, 3, Default::default())) // Adjusted for 4 input channels
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| Tensor::max_pool2d(xs, 2, 2, 0, 1, false))
        .add(nn::conv2d(vs / "conv2", 16, 32, 3, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| Tensor::max_pool2d(xs, 2, 2, 0, 1, false))
        .add_fn(|xs| xs.view([-1, 32 * 54 * 54])) // Assuming output of conv layers is [batch_size, 32, 54, 54]
        .add(nn::linear(vs / "fc1", 32 * 54 * 54, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "fc2", 128, 2, Default::default()));
    net
}
 */

fn train_network(net: &mut nn::Sequential,
                 optimizer: &mut nn::Optimizer,
                 train_loader: &Vec<(Tensor, Tensor)>,
                 num_epochs: i32) {
    for epoch in 1..=num_epochs {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total = 0;

        for (images, labels) in train_loader.iter() {
            let predictions = net.forward(&images);
            let loss = predictions.cross_entropy_for_logits(&labels);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += f64::from(loss.sum(Kind::Float).double_value(&[]));
            total_correct += label_accuracy(&predictions, &labels);
            total += labels.size()[0];
        }

        println!("Epoch: {}, Loss: {:.4}, Accuracy: {:.4}", epoch, total_loss / total as f64, total_correct as f64 / total as f64);
    }
}

fn load_image(image: &DynamicImage) -> Result<Tensor, Box<dyn Error>> {
    // Ensure the image is of size 224x224
    let resized_image = image.resize_exact(224, 224, image::imageops::FilterType::Nearest);

    // Convert the DynamicImage to a Vec<u8> in RGB format
    let rgb_pixels: Vec<u8> = resized_image.to_rgb8().into_raw();

    // Convert the Vec<u8> to a Vec<f32> and normalize to [0.0, 1.0]
    let img_floats: Vec<f32> = rgb_pixels.iter().map(|&pixel| pixel as f32 / 255.0).collect();

    // Check if the img_floats has the correct number of elements
    if img_floats.len()!= 224 * 224 * 3 {
        return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "Image has unexpected number of elements")));
    }

    // Use f_from_slice to create a tensor from the float slice
    let img_tensor = Tensor::f_from_slice(&img_floats)?;

    // Reshape the tensor to [batch_size, num_channels, height, width]
    // Assuming batch_size is 1 since we're loading a single image
    let img_tensor = img_tensor.reshape([1, 3, 224, 224]);

    Ok(img_tensor)
}

fn dropout(p: f64, train: bool) -> impl Fn(&Tensor) -> Tensor + 'static {
    move |input| input.dropout(p, train)
}

fn create_network(vs: &nn::Path) -> nn::Sequential {
    let net = nn::seq()
        .add(nn::conv2d(vs / "conv1", 3, 16, 3, Default::default())) // Initial convolutional layer
        .add_fn(|xs| xs.relu()) // ReLU activation
        .add_fn(|xs| Tensor::max_pool2d(xs, 2, 2, 0, 1, false)) // Max pooling
        .add_fn(dropout(0.25, true)) // Dropout layer to prevent overfitting
        .add(nn::conv2d(vs / "conv2", 16, 32, 3, Default::default())) // Second convolutional layer
        .add_fn(|xs| xs.relu()) // ReLU activation
        .add_fn(|xs| Tensor::max_pool2d(xs, 2, 2, 0, 1, false)) // Max pooling
        .add_fn(dropout(0.25, true)) // Another dropout layer
        .add(nn::conv2d(vs / "conv3", 32, 64, 3, Default::default())) // Third convolutional layer
        .add_fn(|xs| xs.relu()) // ReLU activation
        .add_fn(|xs| Tensor::max_pool2d(xs, 2, 2, 0, 1, false)) // Max pooling
        .add_fn(dropout(0.25, true)) // Dropout layer
        .add_fn(|xs| xs.view([-1, 64 * 26 * 26])) // Corrected Flatten the tensor for the fully connected layer
        .add(nn::linear(vs / "fc1", 64 * 26 * 26, 128, Default::default())) // First fully connected layer
        .add_fn(|xs| xs.relu()) // ReLU activation
        .add_fn(dropout(0.5, true)) // High dropout rate for the fully connected layer
        .add(nn::linear(vs / "fc2", 128, 2, Default::default())); // Output layer

    net
}

fn evaluate_network(net: &nn::Sequential,
                    test_loader: &Vec<(Tensor, Tensor)>) -> f64 {
    let mut total_correct = 0;
    let mut total = 0;

    for (images, labels) in test_loader.iter() {
        let predictions = net.forward(&images);
        total_correct += label_accuracy(&predictions, &labels);
        total += labels.size()[0];
    }

    total_correct as f64 / total as f64
}

fn label_accuracy(predictions: &Tensor, labels: &Tensor) -> i64 {
    let predicted_labels = predictions.argmax(-1, false);
    let correct_predictions = predicted_labels.eq_tensor(&labels);

    correct_predictions.sum(Kind::Int64).int64_value(&[])
}

fn main() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let net = &mut create_network(&vs.root());
    let mut optimizer = nn::Adam::default().build(&vs, 0.00000001).unwrap();

    let data_dir = "hymenoptera_data";

    let train_dataset = ImageDataset::new(data_dir, &["train"]);
    let test_dataset = ImageDataset::new(data_dir, &["val"]);

    let train_loader = data_loader(&train_dataset, 4, true);
    let test_loader = data_loader(&test_dataset, 4, false);

    let num_epochs = 15;

    let start_time = Instant::now();
    train_network(net, &mut optimizer, &train_loader, num_epochs);
    let elapsed = start_time.elapsed();
    println!("Training time: {:?}", elapsed);

    let test_accuracy = evaluate_network(net, &test_loader);
    println!("Test accuracy: {:.4}", test_accuracy);
}
