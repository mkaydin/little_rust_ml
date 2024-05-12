use std::{collections::HashMap};
use tch::{Tensor, Kind, Device, vision, Scalar};

// defining neural network 2 linear layer
struct MyModel {
    l1: Linear,
    l2: Linear,
}

impl MyModel {
    fn new (mem: &mut Memory) -> MyModel {
        let l1 = Linear::new(mem, 784, 128);
        let l2 = Linear::new(mem, 128, 10);
        Self {
            l1: l1,
            l2: l2,
        }
    }
}

impl Compute for MyModel {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let mut o = self.l1.forward(mem, input);
        o = o.relu();
        o = self.l2.forward(mem, &o);
        o
    }
}

// defining neural network layer, a linear layer
trait Compute {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor;
}

struct Linear {
    params: HashMap<String, usize>,
}

impl Linear {
    fn new (mem: &mut Memory, ninputs: i64, noutputs: i64) -> Self {
        let mut p = HashMap::new();
        p.insert("W".to_string(), mem.new_push(&[ninputs,noutputs], true));
        p.insert("b".to_string(), mem.new_push(&[1, noutputs], true));

        Self {
            params: p,
        }
    }
}

impl Compute for Linear {
    fn forward (&self,  mem: &Memory, input: &Tensor) -> Tensor {
        let w = mem.get(self.params.get(&"W".to_string()).unwrap());
        let b = mem.get(self.params.get(&"b".to_string()).unwrap());
        input.matmul(w) + b
    }
}

// mean squared error and cross entropy loss function
fn mse(target: &Tensor, pred: &Tensor) -> Tensor {
    (target - pred).square().mean(Kind::Float)
}

fn cross_entropy (target: &Tensor, pred: &Tensor) -> Tensor {
    let loss = pred.log_softmax(-1, Kind::Float).nll_loss(target);
    loss
}

// the tensor store memory
struct Memory {
    size: usize,
    values: Vec<Tensor>,
}

impl Memory {

    fn new() -> Self {
        let v = Vec::new();
        Self {size: 0,
            values: v}
    }

    fn push (&mut self, value: Tensor) -> usize {
        self.values.push(value);
        self.size += 1;
        self.size-1
    }

    fn new_push (&mut self, size: &[i64], requires_grad: bool) -> usize {
        let t = Tensor::randn(size, (Kind::Float, Device::Cpu)).requires_grad_(requires_grad);
        self.push(t)
    }

    fn get (&self, addr: &usize) -> &Tensor {
        &self.values[*addr]
    }

    fn apply_grads_sgd(&mut self, learning_rate: f32) {
        let mut g = Tensor::new();
        self.values
            .iter_mut()
            .for_each(|t| {
                if t.requires_grad() {
                    g = t.grad();
                    t.set_data(&(t.data() - learning_rate*&g));
                    t.zero_grad();
                }
            });
    }

    fn apply_grads_sgd_momentum(&mut self, learning_rate: f32) {
        let mut g: Tensor = Tensor::new();
        let mut velocity: Vec<Tensor>= Tensor::zeros(&[self.size as i64], (Kind::Float, Device::Cpu)).split(1, 0);
        let mut vcounter = 0;
        const BETA:f32 = 0.9;

        self.values
            .iter_mut()
            .for_each(|t| {
                if t.requires_grad() {
                    g = t.grad();
                    velocity[vcounter] = BETA * &velocity[vcounter] + (1.0 - BETA) * &g;
                    t.set_data(&(t.data() - learning_rate * &velocity[vcounter]));
                    t.zero_grad();
                }
                vcounter += 1;
            });
    }
}

// the training loop
fn train<F>(mem: &mut Memory, x: &Tensor, y: &Tensor, model: &dyn Compute, epochs: i64, batch_size: i64, errfunc: F, learning_rate: f32)
    where F: Fn(&Tensor, &Tensor)-> Tensor
{
    let mut error = Tensor::from(0.0);
    let mut batch_error = Tensor::from(0.0);
    let mut pred = Tensor::from(0.0);
    for epoch in 0..epochs {
        batch_error = Tensor::from(0.0);
        for (batchx, batchy) in get_batches(&x, &y, batch_size, true) {
            pred = model.forward(mem, &batchx);
            error = errfunc(&batchy, &pred);
            batch_error += error.detach();
            error.backward();
            mem.apply_grads_sgd_momentum(learning_rate);
        }
        println!("Epoch: {:?} Error: {:?}", epoch, batch_error/batch_size);
    }
}

// mini batching
fn get_batches(x: &Tensor, y: &Tensor, batch_size: i64, shuffle: bool) -> impl Iterator<Item = (Tensor, Tensor)> {
    let num_rows = x.size()[0];
    let num_batches = (num_rows + batch_size - 1) / batch_size;

    let indices = if shuffle {
        Tensor::randperm(num_rows as i64, (Kind::Int64, Device::Cpu))
    } else
    {
        let rng = (0..num_rows).collect::<Vec<i64>>();
        Tensor::from_slice(&rng)
    };
    let x = x.index_select(0, &indices);
    let y = y.index_select(0, &indices);

    (0..num_batches).map(move |i| {
        let start = i * batch_size;
        let end = (start + batch_size).min(num_rows);
        let batchx: Tensor = x.narrow(0, start, end - start);
        let batchy: Tensor = y.narrow(0, start, end - start);
        (batchx, batchy)
    })
}

// load data set
fn load_mnist() -> (Tensor, Tensor) {
    let m = vision::mnist::load_dir("data").unwrap();
    let x = m.train_images;
    let y = m.train_labels;
    (x, y)
}

// calculate accuracy
fn accuracy(target: &Tensor, pred: &Tensor) -> f64 {
    let yhat = pred.argmax(1,true).squeeze();
    let eq = target.eq_tensor(&yhat);
    let accuracy: f64 = (eq.sum(Kind::Int64) / target.size()[0]).double_value(&[]).into();
    accuracy
}

// instanting and training neural network model
fn main() {
    let (x, y) = load_mnist();

    let mut m = Memory::new();
    let mymodel = MyModel::new(&mut m);
    train(&mut m, &x, &y, &mymodel, 100, 128, cross_entropy, 0.3);
    let out = mymodel.forward(&m, &x);
    println!("Training Accuracy: {}", accuracy(&y, &out));
}
