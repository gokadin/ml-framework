package datasets

import (
	"log"
)

const datasetNameMNIST = "mnist"
const datasetNameXor = "xor"

func From(source string) *Dataset {
	switch source {
	// file formats
	default:
		return fromNamed(source)
	}
}

func fromNamed(datasetName string) *Dataset {
	createCache()

	switch datasetName {
	case datasetNameMNIST:
		return loadMNIST()
	case datasetNameXor:
		return loadXor()
	default:
		log.Fatalf("Could not find Dataset with name %s", datasetName)
	}

	return &Dataset{}
}

func FromData(data []float32) *Dataset {
	return &Dataset{}
}