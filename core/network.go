package core

import (
	"github.com/gokadin/ml-framework/layer"
	"github.com/gokadin/ml-framework/tensor"
	"log"
	"math/rand"
	"time"
)

type Network struct {
	builder *builder
	layers  []*layer.Layer
}

func NewNetwork() *Network {
	rand.Seed(time.Now().UTC().UnixNano())

	n := &Network{
		layers: make([]*layer.Layer, 0),
	}

	n.builder = newBuilder(n)

	return n
}

func (n *Network) AddInputLayer(size int) *builder {
	n.builder.addInputLayer(size)
	return n.builder
}

func (n *Network) LayerCount() int {
	return len(n.layers)
}

func (n *Network) GetLayers() []*layer.Layer {
	return n.layers
}

func (n *Network) GetLayer(index int) *layer.Layer {
	if index < 0 || index > n.LayerCount() - 1 {
		log.Fatal("requested layer at index", index, "does not exist")
	}

	return n.layers[index]
}

func (n *Network) InputLayer() *layer.Layer {
	if n.LayerCount() == 0 {
		log.Fatal("input layer not set")
	}

	return n.layers[0]
}

func (n *Network) OutputLayer() *layer.Layer {
	if n.LayerCount() == 0 || !n.layers[n.LayerCount() - 1].IsOutputLayer() {
		log.Fatal("output layer not set")
	}

	return n.layers[n.LayerCount() - 1]
}

func (n *Network) Forward(input *tensor.Tensor) *tensor.Tensor {
	return n.InputLayer().Forward(input)
}
