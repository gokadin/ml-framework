package datasets

import (
	"log"
	"ml-framework/mat"
)

const xorName = "xor"

func loadXor() *Dataset {
	log.Print("Getting XOR Dataset...")

	dataset := NewDataset()
	dataset.SetName(xorName)
	dataset.Set(TrainingSetX).SetData(mat.FromSlice32f(mat.Dim(4, 2), []float32{1, 0, 1, 1, 0, 1, 0, 0}))
	dataset.Set(TrainingSetY).SetData(mat.FromSlice32f(mat.Dim(4, 1), []float32{1, 0, 1, 0}))
	dataset.Set(ValidationSetX).SetData(mat.FromSlice32f(mat.Dim(4, 2), []float32{1, 0, 1, 1, 0, 1, 0, 0}))
	dataset.Set(ValidationSetY).SetData(mat.FromSlice32f(mat.Dim(4, 1), []float32{1, 0, 1, 0}))
	return dataset
}
