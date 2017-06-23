import tensorpack as tp

def get_cifar(train_or_test, batch_size=None):
    # Get CIFAR data generator
    df = tp.dataset.Cifar10(train_or_test)
    if batch_size:
        df = tp.BatchData(df, batch_size)
    df.reset_state()
    ds = df.get_data()
    return ds

def get_examples(n_examples):
    ds = get_cifar('test')
    # Get 10 examples from each class
    # Generator 'ds' performs randomisation of examples
    examples = [[] for x in xrange(n_examples)]
    for d in ds:
        # Add if this class does not have 10 examples already
        if len(examples[d[1]]) < n_examples:
            examples[d[1]].append(d[0])
            # Break when all classes have 10 examples
            if all(map(lambda x: len(x) == n_examples, examples)):
                break
    return examples