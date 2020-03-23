# **ml-framework** is a petty attempt at a machine learning framework from scratch in Go

This project aims to build a framework for training and running artificial neural networks using tensors and dynamic computational graphs. 

Currently the main purpose is for learning how to implement each of a major framework's (such as [TensorFlow](https://www.tensorflow.org/)) component from scratch. 

## Documentation

[https://github.com/gokadin/ml-framework/wiki](https://github.com/gokadin/ml-framework/wiki)

## Features

- **mat** package with common vector and matrix operations. 
- **tensor** package with dynamic computational graph building. 
- **dataset** package for automatically getting popular datasets from the net or for processing local files. 
- **modules** package implementing the most common ANN layers and activation functions. 
- **models** package for building ANN models and running/training them. 
- GPU support on the way... 

## Example

The following demonstrates how to train the mnist digits dataset. 

```go
// get the dataset
dataset := datasets.From("mnist").SetBatchSize(1000)

// build the model
model := models.Build(
	modules.Dense(128, modules.ActivationRelu),
	modules.Dense(10, modules.ActivationSoftmax))

// configure the model
model.Configure(models.ModelConfig{
	Epochs: 3,
	Loss: models.LossSoftmaxCrossEntropy,
})

// run and validate the model
model.Fit(dataset)
model.Run(dataset)
```

## Contributing

This is a personal project with the aim of learning the ANN world and is not intended to be used in any production application. 

However, any contribution is welcome as it will most likely teach someone something new. Feel free to open a PR anytime. 

You can also contact me for any questions on how to contribute or just for general comments about the project. 