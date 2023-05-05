def get_parameter(dataset):
    if dataset == 'usps': # 0.050324, 1, 1e-7 # DNystroem 0.062282, 10, 1e-6
        parameter_dic = { 
            'task_type' : 'multiclass',
            'num_examples' : 7291,
            'm' : 20,
            'M' : 400,
            'num_test' : 2007,
            'dimensional' : 256,
            'num_classes' : 10,
            'kernel_type' : 'gaussian',
            'kernel_par' : 10,
            'lambda_reg' : 1e-6,
        }
    elif dataset == 'pendigits': # 0.0211, 100, 1e-6 # Dynstroem 0.026301, 100, 1e-5
        parameter_dic = { 
            'task_type' : 'multiclass',
            'num_examples' : 7494,
            'm' : 20,
            'M' : 400,
            'num_test' : 3498,
            'dimensional' : 16,
            'num_classes' : 10,
            'kernel_type' : 'gaussian',
            'kernel_par' : 100,
            'lambda_reg' : 1e-6,
        }
    elif dataset == 'letter':  # 0.0398, 0.1, 1e-6 # Dynstroem 0.0556, 1, 1e-7
        parameter_dic = {
            'task_type' : 'multiclass',
            'num_examples' : 15000,
            'num_test' : 5000,
            'M': 800,
            'm': 30,
            'dimensional' : 16,
            'num_classes' : 26,
            'kernel_type' : 'gaussian',
            'kernel_par' : 1,
            'lambda_reg' : 1e-7,
        }
    elif dataset == 'mnist': # 0.0286
        parameter_dic = {
            'task_type' : 'multiclass',
            'num_examples' : 60000,
            'num_test' : 10000,
            'dimensional' : 784,
            'num_classes' : 10,
            'kernel_type' : 'gaussian',
            'kernel_par' : 10, 
            'lambda_reg' : 1e-6
        }
    else:
        parameter_dic = {
            'task_type' : 'multiclass',
            'num_classes' : 10,
            'dimensional' : 784,
            'kernel_type' : 'gaussian',
            'kernel_par' : 0.1, 
            'lambda_reg' : 0.00001
        }
    return parameter_dic
