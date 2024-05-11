use std::{collections::HashMap};
use tch::{Tensor, Kind, Device, vision, Scalar};

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
    fn forward (&self, mem: &Memory, input: &Tensor) -> Tensor {
        let mut o = self.l1.forward(mem, input);
        o = o.relu();
        o = self.l2.forward(mem, &o);
    }
}

fn main() {
    println!("Hello, world!");
}
