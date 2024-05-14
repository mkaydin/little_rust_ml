use tch::{Tensor, vision::image, Kind, Device};

fn rgb_to_grayscale(tensor: &Tensor) -> Tensor {
    let red_channel = tensor.get(0);
    let green_channel = tensor.get(1);
    let blue_channel = tensor.get(2);

    // calculate the grayscale tensor using the luminance formula
    let grayscale = (red_channel * 0.2989) +  (green_channel * 0.5870) + (blue_channel * 0.1140);
    grayscale.unsqueeze(0)
}

pub fn apply_kernels() {
    let mut img = image::load("mypic.jpg").expect("Error loading image");
    img = rgb_to_grayscale(&img).reshape(&[1,1,1024,1024]);
    let bias: Tensor = Tensor::full(&[1], 0.0, (Kind::Float, Device::Mps));

    // define and apply guassian kernel
    let mut k1 = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    for element in k1.iter_mut() {
        *element /= 16.0;
    }

    let kernel1 = Tensor::from_slice(&k1).reshape(&[1,1,3,3]).to_kind(Kind::Float);

    img = img.conv2d(&kernel1, Some(&bias), &[1], &[0], &[1],1);

    // define and apply laplacian Kernel
    let k2 = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let kernel2 = Tensor::from_slice(&k2).reshape(&[1,1,3,3]).to_kind(Kind::Float);

    img = img.conv2d(&kernel2, Some(&bias), &[1], &[0], &[1], 1);

    image::save(&img, "filtered.jpg").expect("Couldn't save the filtered img");
}