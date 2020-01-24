package datasets

import "log"

const TrainingSetX = "trainingSetX"
const TrainingSetY = "trainingSetY"
const ValidationSetX = "validationSetX"
const ValidationSetY = "validationSetY"

type Dataset struct {
	name string
	sets map[string]*set
}

func NewDataset() *Dataset {
	return &Dataset{
		sets: make(map[string]*set),
	}
}

func (d *Dataset) SetName(name string) {
	d.name = name
}

func (d *Dataset) AddData(name string, data [][]float64) {
	d.sets[name] = &set{data}
}

func (d *Dataset) Get(name string) *set {
	if set, ok := d.sets[name]; ok {
		return set
	}

	log.Fatalf("set %s does not exist", name)
	return nil
}

func (d *Dataset) InputSize() int {
	if set, ok := d.sets[TrainingSetX]; ok {
		return len(set.data)
	}

	return 0
}