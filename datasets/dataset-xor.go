package datasets

import "log"

const xorName = "xor"

func loadXor() *Dataset {
	log.Print("Getting XOR Dataset...")

	dataset := NewDataset()
	dataset.SetName(xorName)
	dataset.AddData(TrainingSetX, [][]float64{{1, 0}, {1, 1}, {0, 1}, {0, 0}})
	dataset.AddData(TrainingSetY, [][]float64{{1}, {0}, {1}, {0}})
	dataset.AddData(ValidationSetX, [][]float64{{1, 0}, {1, 1}, {0, 1}, {0, 0}})
	dataset.AddData(ValidationSetY, [][]float64{{1}, {0}, {1}, {0}})
	return dataset
}
