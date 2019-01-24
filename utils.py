import os

import tensorflow as tf


def check_folders(paths):
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tensor_description(var):
    """Returns a compact and informative string about a tensor.
    Args:
        var: A tensor variable.
    Returns:
        a string with type and size, e.g.: (float32 1x8x8x1024).
    """
    description = '(' + str(var.dtype.name) + ' '
    sizes = var.get_shape()
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += 'x'
    description += ')'
    return description


def analyze_vars(variables, path):
    """Prints the names and shapes of the variables.
    Args:
        variables: list of variables, for example tf.global_variables().
        print_info: Optional, if true print variables and their shape.
    Returns:
        (total size of the variables, total bytes of the variables)
    """
    f = open(path, 'w')
    f.write('---------\n')
    f.write('Variables: name (type shape) [size]\n')
    f.write('---------\n')
    total_size = 0
    total_bytes = 0
    for var in variables:
        # if var.num_elements() is None or [] assume size 0.
        var_size = var.get_shape().num_elements() or 0
        var_bytes = var_size * var.dtype.size
        total_size += var_size
        total_bytes += var_bytes
        f.write(var.name+' '+tensor_description(var)+' '+'[%d, bytes: %d]\n' % (var_size, var_bytes))
    f.write('Total size of variables: %d\n' % total_size)
    f.write('Total bytes of variables: %d\n' % total_bytes)
    return total_size, total_bytes