## Generate datasets for machine learning classification tasks, the Tao way.

#### Introduction and Motivation

    data, labels = yin_yang_datagen(n = 3000, num_target_classes = 3)

![](yin_yang_images/intro.PNG)

As learners and practitioners of data science, we seek to understand the underlying workings of complex, "black box" algorithms. There is no better way to understand the optimization methods employed by the algorithm to fit a training dataset than by visualizing the progress of the fit over time (training epochs). For certain datasets, we also want to understand just <i>which</i> classification methods perform better than others and why. The `yin_yang_dataset` generates datasets of shape `(m, 2)`, with `m` being the desired size, and corresponding target values for use by classification algorithms.

    data.shape = (3000,2)
    labels.shape = (3000,)

#### Basic Usage:

    from yin_yang import yin_yang_datagen
    data, labels = yin_yang_datagen(n = 1500, size_ratio = 0.18, num_target_classes = 2)
    
    def cmapper(x):
        if x == 1:
            return 'red'
        if x == -1:
            return 'blue'
        if x == 0:
            return 'orange'
    
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], s = 5, c = list(map(cmapper, labels)))


    






