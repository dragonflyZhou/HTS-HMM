This is a python(v3.4.1) implementation of HTS-2.2_for_HTK-3.4.1(http://hts.sp.nitech.ac.jp/?Home).

The intention is to help people understand the algorithm better and 
for me personally a practice in python. 

Supported:
1. Two training methods, viterbi and balm-welch are implemented. 
2. MSD(multi-space probability) is supported.

Not supported yet:
1. Joint duration training 
2. Two or more mixtures per stream
3. Tied models
4. full covariance matrix

testViterbi.py is a script generates a sequence of observations from test.mmf, 
then runs viterbi for one iteration using the same test.mmf as the starting point.
The viterbi path is the same as the true states, and the log probability matches with the true value.

verifyBaumWelch.py is a script processes files from HTS-demo_CMU-ARCTIC-SLT dataset. 
Monophone models are verified by the author. 

Have fun!
Jingting Zhou