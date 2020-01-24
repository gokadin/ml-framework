package datasets

import (
	"log"
)

const datasetNameMNIST = "mnist"

func From(source string) *Dataset {
	switch source {
	// file formats
	default:
		return fromOnline(source)
	}
}

func fromOnline(datasetName string) *Dataset {
	createCache()

	switch datasetName {
	case datasetNameMNIST:
		return loadMNIST()
	default:
		log.Fatalf("Could not find Dataset with name %s", datasetName)
	}

	return &Dataset{}
}