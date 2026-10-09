Basic Example
==============

Important: 
Be sure to initialize Kokkos before setting up any objects from this library,
by created a scoped KokkosParser object, i.e:

.. code-block:: python

    import numpy as np
    src_coords = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1],[2,-1],[2,0],[2,1]], dtype='f8')
    trg_coords = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]], dtype='f8')
    data       = np.array([[-1],[-1],[-1],[0],[0],[0],[1],[1],[1]], dtype='f8')
    
    import pycompadre
    
    # initialize kokkos
    kp = pycompadre.KokkosParser()
    
    # parameters
    poly_order = 3
    dimension = 2
    epsilon_multiplier = 1.5
    number_of_batches = 2
    
    # set up GMLS class
    gmls_obj = pycompadre.GMLS(poly_order, dimension)
    # set up ParticleHelper class
    gmls_helper = pycompadre.ParticleHelper(gmls_obj)
    
    # build Kd-Tree
    gmls_helper.generateKDTree(src_coords)
    # do neighbor search
    gmls_helper.generateNeighborListsFromKNNSearchAndSet(trg_coords, poly_order, dimension, epsilon_multiplier)
    
    # set target operation
    gmls_obj.addTargets(pycompadre.TargetOperation.ScalarPointEvaluation)
    # generate stencil with number of batches set to 2, and not keeping coefficients
    gmls_obj.generateAlphas(number_of_batches, keep_coefficients=False)
    
    # apply stencil to sample data for all targets
    computed_point_values = gmls_helper.applyStencil(data, pycompadre.TargetOperation.ScalarPointEvaluation)
    
    # clean up objects in order of dependency
    del gmls_obj
    del gmls_helper
    del kp

