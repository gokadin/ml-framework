package data

import (
	"github.com/gokadin/ml-framework/tensor"
)

type Dataset struct {
	builder *builder
    data *tensor.Tensor
}

func NewDataset() *Dataset {
	dataset := &Dataset{}
	dataset.builder = newBuilder(dataset)
	return dataset
}

func (d *Dataset) FromCsv(filename string, startIndex, endIndex, limit int) *builder {
	return d.builder.readCsv(filename, startIndex, endIndex, limit)
}

func (d *Dataset) FromRandom(associations, size int) *builder {
	return d.builder.readRandom(associations, size)
}

func (d *Dataset) Data() *tensor.Tensor {
	return d.data
}

func (d *Dataset) Size() int {
	return len(d.data.Data())
}
