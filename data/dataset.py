from .data_loaders import *


class Data(object):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.pointer = 0

    def normalize_sample(self, x):
        x = x - x.mean(axis=(1, 2, 3), keepdims=True)
        x = x / x.std(axis=(1, 2, 3), keepdims=True)
        return x

    def next_batch(self, bs, shuffle=True, normalize=False):
        if shuffle:
            idx = np.random.choice(len(self.images), bs, replace=False)
        else:
            idx = range(self.pointer, self.pointer + bs)
            self.pointer += bs
            if self.pointer + bs > len(self.images):
                self.pointer = 0
        x = self.images[idx]
        if normalize:
            x = self.normalize_sample(x)
        y = self.labels[idx]
        return x, y


class usps(object):
    def __init__(self, FLAGS):
        usps_data = load_usps(val=FLAGS.val, scale28=True, zero_centre=FLAGS.zc)
        train_x = usps_data.train_X
        test_x = usps_data.test_X
        val_x = usps_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[usps_data.train_y.reshape(-1)]
        test_y = np.eye(10)[usps_data.test_y.reshape(-1)]
        val_y = np.eye(10)[usps_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class mnist(object):
    def __init__(self, FLAGS):
        mnist_data = load_mnist(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = mnist_data.train_X
        test_x = mnist_data.test_X
        val_x = mnist_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[mnist_data.train_y.reshape(-1)]
        test_y = np.eye(10)[mnist_data.test_y.reshape(-1)]
        val_y = np.eye(10)[mnist_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class mnistm(object):
    def __init__(self, FLAGS):
        train, test, val = load_mnistm(os.path.join(FLAGS.datadir, 'mnistm'), val=FLAGS.val, zero_centre=FLAGS.zc)
        train_x = train[0]
        train_y = train[1]
        test_x = test[0]
        test_y = test[1]
        val_x = val[0]
        val_y = val[1]
        """
        train_x = np.transpose(train_x, (0, 3, 1, 2))
        test_x = np.transpose(test_x, (0, 3, 1, 2))
        val_x = np.transpose(val_x, (0, 3, 1, 2))
        """
        train_y = np.eye(10)[train_y.reshape(-1)]
        test_y = np.eye(10)[test_y.reshape(-1)]
        val_y = np.eye(10)[val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class svhn(object):
    def __init__(self, FLAGS):
        svhn_data = load_svhn(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = svhn_data.train_X
        test_x = svhn_data.test_X
        val_x = svhn_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[svhn_data.train_y.reshape(-1)]
        test_y = np.eye(10)[svhn_data.test_y.reshape(-1)]
        val_y = np.eye(10)[svhn_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class syndigits(object):
    def __init__(self, FLAGS):
        syndigits_data = load_syn_digits(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = syndigits_data.train_X
        test_x = syndigits_data.test_X
        val_x = syndigits_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(10)[syndigits_data.train_y.reshape(-1)]
        test_y = np.eye(10)[syndigits_data.test_y.reshape(-1)]
        val_y = np.eye(10)[syndigits_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class cifar(object):
    def __init__(self, FLAGS):
        cifar_data = load_cifar10(val=FLAGS.val, range_01=FLAGS.zc)

        train_x = cifar_data.train_X
        test_x = cifar_data.test_X
        val_x = cifar_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(9)[cifar_data.train_y.reshape(-1)]
        test_y = np.eye(9)[cifar_data.test_y.reshape(-1)]
        val_y = np.eye(9)[cifar_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class stl(object):
    def __init__(self, FLAGS):
        stl_data = load_stl(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = stl_data.train_X
        test_x = stl_data.test_X
        val_x = stl_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(9)[stl_data.train_y.reshape(-1)]
        test_y = np.eye(9)[stl_data.test_y.reshape(-1)]
        val_y = np.eye(9)[stl_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class gtsrb(object):
    def __init__(self, FLAGS):
        gtsrb_data = load_gtsrb(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = gtsrb_data.train_X
        test_x = gtsrb_data.test_X
        val_x = gtsrb_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(43)[gtsrb_data.train_y.reshape(-1)]
        test_y = np.eye(43)[gtsrb_data.test_y.reshape(-1)]
        val_y = np.eye(43)[gtsrb_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


class synsigns(object):
    def __init__(self, FLAGS):
        synsigns_data = load_syn_signs(val=FLAGS.val, zero_centre=FLAGS.zc)

        train_x = synsigns_data.train_X
        test_x = synsigns_data.test_X
        val_x = synsigns_data.val_X

        train_x = np.transpose(train_x, (0, 2, 3, 1)).astype(np.float32)
        test_x = np.transpose(test_x, (0, 2, 3, 1)).astype(np.float32)
        val_x = np.transpose(val_x, (0, 2, 3, 1)).astype(np.float32)

        train_y = np.eye(43)[synsigns_data.train_y.reshape(-1)]
        test_y = np.eye(43)[synsigns_data.test_y.reshape(-1)]
        val_y = np.eye(43)[synsigns_data.val_y.reshape(-1)]

        self.train = Data(train_x, train_y)
        self.test = Data(test_x, test_y)
        self.val = Data(val_x, val_y)

        self.train_num = train_y.shape[0]
        self.test_num = test_y.shape[0]
        self.val_num = val_y.shape[0]


# TODO: Should implement office dataset; amazon, webcam, dslr

def get_attr(source, target):
    # Processed_attr: [processed size, processed channels, number of classes]
    processed_attr = {
        'usps'     : [28, 1, 10],
        'mnist'    : [28, 1, 10],
        'mnistm'   : [28, 3, 10],
        'svhn'     : [32, 3, 10],
        'syndigits': [32, 3, 10],
        'cifar'    : [32, 3, 9],
        'stl'      : [32, 3, 9],
        'gtsrb'    : [40, 3, 43],
        'synsigns' : [40, 3, 43],
        'amazon'   : [227, 3, 31],
        'webcam'   : [227, 3, 31],
        'dslr'     : [227, 3, 31]
    }

    # Desired_attr: [desired size, desired channels, source normalize, target normalize]
    experiment_attr = {
        'usps_mnist'    : [28, 1],
        'mnist_usps'    : [28, 1],
        'mnist_mnistm'  : [28, 3],
        'mnistm_mnist'  : [28, 3],
        'svhn_mnist'    : [32, 1],
        'mnist_svhn'    : [32, 1],
        'svhn_syndigits': [32, 3],
        'syndigits_svhn': [32, 3],
        'cifar_stl'     : [32, 3],
        'stl_cifar'     : [32, 3],
        'synsigns_gtsrb': [40, 3],
        'gtsrb_synsigns': [40, 3],
        'amazon_webcam' : [227, 3],
        'webcam_amazon' : [227, 3],
        'amazon_dslr'   : [227, 3],
        'dslr_amazon'   : [227, 3],
        'webcam_dslr'   : [227, 3],
        'dslr_webcam'   : [227, 3]
    }

    exp = source + '_' + target
    if not exp in experiment_attr:
        raise NotImplementedError("The submitted {} experiment is not valid".format(exp))

    src_sz, src_ch, src_nc = processed_attr[source]
    trg_sz, trg_ch, trg_nc = processed_attr[target]
    exp_sz, exp_ch = experiment_attr[exp]

    return src_sz, trg_sz, exp_sz, src_ch, trg_ch, exp_ch, src_nc


def get_data(domain, FLAGS):
    return eval(domain)(FLAGS)
