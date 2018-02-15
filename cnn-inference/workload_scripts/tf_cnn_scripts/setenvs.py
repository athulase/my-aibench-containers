from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
class arglist :
    cpu = 'bdw'
    model = 'alexnet'
    data_dir = None
    num_omp_threads = None
def setenvs(inpargv):
    args = arglist()
    for i in range(0,len(inpargv)-1) :
        if inpargv[i] == '--cpu' :
            args.cpu = inpargv[i+1]
        elif inpargv[i] == '--model' :            
            args.model = inpargv[i+1]
        elif inpargv[i] == '--data_dir' :            
            args.data_dir = inpargv[i+1]
        elif inpargv[i] == '--num_omp_threads' : 
            args.num_omp_threads = inpargv[i+1]     
    assert (args.cpu == 'knl' or args.cpu == 'bdw' or args.cpu == 'skl' or args.cpu == 'knm')
    if (args.cpu == 'bdw' and args.model == 'alexnet') :
        os.environ["KMP_BLOCKTIME"] = "30"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    elif (args.cpu == 'knl' and args.model == 'alexnet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads 
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"] = "66"
        else:
          os.environ["OMP_NUM_THREADS"]= "136"        
    elif (args.cpu == 'skl' and args.model == 'alexnet') :
        os.environ["KMP_BLOCKTIME"] = "30"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    elif (args.cpu == 'knm' and args.model == 'alexnet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"] = "36"
        else:
          os.environ["OMP_NUM_THREADS"]= "144" 
    elif (args.cpu == 'bdw' and args.model == 'googlenet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"
    elif (args.cpu == 'knl' and args.model == 'googlenet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads 
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "66"
        else:
          os.environ["OMP_NUM_THREADS"]= "66"
    elif (args.cpu == 'skl' and args.model == 'googlenet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"
        os.environ["OMP_PROC_BIND"] = "true"
        os.environ["OMP_NUM_THREADS"]= "56"
    elif (args.cpu == 'knm' and args.model == 'googlenet'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "36"
        else:
          os.environ["OMP_NUM_THREADS"]= "72"

    elif (args.cpu =='bdw' and args.model == 'vgg11'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    elif (args.cpu =='knl' and args.model == 'vgg11'):    
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads 
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "66"
        else:
          os.environ["OMP_NUM_THREADS"]= "68"
    elif (args.cpu == 'skl' and args.model == 'vgg11'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    elif (args.cpu =='knm' and args.model == 'vgg11'):
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "36"
        else:
          os.environ["OMP_NUM_THREADS"]= "72"

    elif (args.cpu =='bdw' and args.model == 'inception3'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"     
    elif (args.cpu =='knl' and args.model == 'inception3'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads 
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "66"
        else:
          os.environ["OMP_NUM_THREADS"]= "66"       
    elif (args.cpu == 'skl' and args.model == 'inception3'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"
        os.environ["OMP_NUM_THREADS"]= "56"
    elif (args.cpu =='knm' and args.model == 'inception3'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "50"
        else:
          os.environ["OMP_NUM_THREADS"]= "72"

    elif (args.cpu =='bdw' and args.model == 'resnet50'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"
    elif (args.cpu =='knl' and args.model == 'resnet50'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads 
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "67"
        else:
          os.environ["OMP_NUM_THREADS"]= "66"    
    elif (args.cpu == 'skl' and args.model == 'resnet50'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"]="granularity=fine,verbose,compact,1,0"
        os.environ["OMP_NUM_THREADS"]= "56"
    elif (args.cpu =='knm' and args.model == 'resnet50'):
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
        if (args.num_omp_threads is not None):
          os.environ["OMP_NUM_THREADS"] = args.num_omp_threads
        elif (args.data_dir is not None):
          os.environ["OMP_NUM_THREADS"]= "50"
        else:
          os.environ["OMP_NUM_THREADS"]= "144"

    # TF_ADJUST_HUE_FUSED will skip expensive data conversion from RGB2HSV and
    # then HSV2RGB. The fused function is NOT implemented on GPU, it is 
    # off by default. Since this benchmark is running on CPU, we should always 
    # turn it on
    os.environ["TF_ADJUST_HUE_FUSED"] = "1"
    
    return args
