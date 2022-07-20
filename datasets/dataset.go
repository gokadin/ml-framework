package datasets

const defaultBatchSize = 1

type Dataset struct {
	name      string
	sets      map[string]*set
	batchSize int
}

func NewDataset() *Dataset {
	return &Dataset{
		sets:      make(map[string]*set),
		batchSize: defaultBatchSize,
	}
}

func (d *Dataset) SetName(name string) {
	d.name = name
}

func (d *Dataset) SetBatchSize(batchSize int) *Dataset {
	d.batchSize = batchSize
	for _, set := range d.sets {
		set.SetBatchSize(batchSize)
	}
	return d
}

func (d *Dataset) Set(name string) *set {
	if _, ok := d.sets[name]; !ok {
		d.sets[name] = newSet(d.batchSize)
	}

	return d.sets[name]
}
