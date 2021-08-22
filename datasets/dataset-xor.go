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
	dataset.AddData(TrainingSetX, mat.NewMat32f(mat.WithShape(4, 2), []float32{1, 0, 1, 1, 0, 1, 0, 0}))
	dataset.AddData(TrainingSetY, mat.NewMat32f(mat.WithShape(4, 1), []float32{1, 0, 1, 0}))
	dataset.AddData(ValidationSetX, mat.NewMat32f(mat.WithShape(4, 2), []float32{1, 0, 1, 1, 0, 1, 0, 0}))
	dataset.AddData(ValidationSetY, mat.NewMat32f(mat.WithShape(4, 1), []float32{1, 0, 1, 0}))
	return dataset
}
