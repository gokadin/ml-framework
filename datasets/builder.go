package datasets

import (
	"log"
)

const datasetNameMNIST = "mnist"

func From(source string) *dataset {
	switch source {
	// file formats
	default:
		return fromOnline(source)
	}
}

func fromOnline(datasetName string) *dataset {
	createCache()

	switch datasetName {
	case datasetNameMNIST:
		return loadMNIST()
	default:
		log.Fatalf("Could not find dataset with name %s", datasetName)
	}

	return &dataset{}
}