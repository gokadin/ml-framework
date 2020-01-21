package datasets

const trainingSetX = "trainingSetX"
const trainingSetY = "trainingSetY"
const validationSetX = "validationSetX"
const validationSetY = "validationSetY"

type dataset struct {
	name string
	sets map[string][][]float64
}

func newDataset() *dataset {
	return &dataset{
		sets: make(map[string][][]float64),
	}
}

func (d *dataset) setName(name string) {
	d.name = name
}

func (d *dataset) addSet(setName string, set [][]float64) {
	d.sets[setName] = set
}
