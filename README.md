# Wigner Distribution Deconvolution with background removal  
# Authors: Oleh Melnyk and Patricia Römer

This code is an implementation of the Wigner Distribution Deconvolution (WDD) algorithm for ptychography. The implementation is based on 

[1] Oleh Melnyk, Phase Retrieval from Short-Time Fourier Measurements and Applications to Ptychography, PhD thesis, Technische Universität München, 2023

and

[2] Oleh Melnyk, Patricia Römer, Background Denoising for Ptychography via Wigner Distribution Deconvolution, arXiv:2402.15353v1

In particular, the folder /figures contains scripts for generating the respective figures from [2]. See them for an example of code usage.

Dependencies: numpy, scipy, matplotlib (/figures), Pillow [PIL] (/figures)

In addition, the repository includes our implementations of

- [adp.py] ADP algorithm based on:

Huibin Chang, Pablo Enfedaque, Jie Zhang, Juliane Reinhardt, Bjoern Enders, Young-Sang Yu, David Shapiro, Christian G. Schroer, Tieyong Zeng, and Stefano Marchesini. Advanced denoising for X-ray ptychography. Optics express, 27(8):10395-10418, 2019.
  
- [prepropcessing.py] background preprocessing proceedure based on:

Chunpeng Wang, Zijian Xu, Haigang Liu, Yong Wang, Jian Wang, and Renzhong Tai. Background noise removal in x-ray ptychography. Applied optics, 56(8):2099–2111, 2017.

Legal Information & Credits
---

Copyright (c) 2024 Oleh Melnyk and Patricia Römer

This software was written by [Oleh Melnyk](https://olehmelnyk.xyz/) and [Patricia Römer](https://www.math.cit.tum.de/math/personen/wissenschaftliches-personal/patricia-roemer/).
It was developed at the Institute of Mathematics at Technical University of Berlin, Institute of Biomedical Imaging at Helmholtz Center Munich and Department of Mathematics at Technical University of Munich.
The authors acknowledge supported by the Helmholtz Association under the contracts No.~ZT-I-0025 (Ptychography 4.0), No.~ZT-I-PF-4-018 (AsoftXm), No.~ZT-I-PF-5-28 (EDARTI), No.~ZT-I-PF-4-024 (BRLEMMM). Patricia Römer acknowledges further financial support by the German Research Foundation (DFG) grants KR 4512/1-1 and KR 4512/2-2.

WDDBackground is free software. You can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. If not stated otherwise, this applies to all files contained in this
package and its sub-directories.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

Description of WDD class
---
The algorithm consists of 3 mains steps and 1 optional step:

Main: 

1) Inversion step
2) Magnitude estimation
3) Phase synchronization

Optional:

1.5) Background Removal    
 
Required parameters:

- ptycho: 

Type: object from forward.py, describes the forward model 

For WDD, the scanning positions ptycho.locations should form a equdistant grid with step ptycho.shift. Use loc_type='grid' for ptycho class. 
Furthermore, the shifts are circular, so that ptycho.circular = True. 
If ptycho.circular = False, the algorithm works and treats noncircular measurements in two possible ways, see parameter add_dummy below. 
If ptycho.shift > 1, then the object to be recovered is assumed to be block-constant, see Section 3.6.5.1 in [1].    

Optional parameters:

- gamma 

Type: integer from 1 to self.par.window_shape[0]//self.par.shift[0] (all diagonals)

Default: all diagonals

The number of diagonals to use for reconstruction. See Assumption A in [2] for details.
 
- reg_type

Type: string, either 'value' or 'percent'

Default: 'value'

- reg_threshold

Type: float  

Default: 0.0

Parameters for regularization of the inversion step by truncation, see Section 3.6.2.2 in [1]. 
When 'value' is chosen, the diagonal's Fourier coefficients corresponding to singular values below reg_threshold are set to zero.
When 'percent' is chosen, the diagonal's Fourier coefficients corresponding to singular values less then quantile(reg_threshold) are set to zero. 

- mg_type

Type: string, either 'diag' or 'log'

Default: 'diag'

Determines, which magnitude estimation method to use, see Section 3.6.3 in [1].

- mg_diagonals_type

Type: string, either 'all', 'value' or 'percent'

Default: 'all'

- mg_diagonals_param

Type: float 

mg_diagonals_type determines whether all diagonals should be used for magnitude estimation. 

If 'value', only mg_diagonals_param of the first diagonals are used, in analogy to gamma. 
If 'percent', only the diagonals where the percentage of non-truncated Fourier coefficients exceeds mg_diagonals_param are used.
For details, see Section 6.1.2.2 in [1].

- as_wtype

Type: string, either 'unweighted','weighted' or 'weighted_sq_amp'

Default: 'weighted'

Choice of weights for phase synchronization. These three choices were discussed in Section 3.6.4 in [1].
Uses mg_diagonals_type, mg_diagonals_param the same way as magnitude estimation in the case memory_saving=True. 

- as_threshold

Type: float 

Default: 10^-10

When constructing the graph for phase syncronization from lifted matrix, its entries below as_threshold 
are treated as zeros. In other words, the corresponding phase differences are not used.

- background 

Type: string, either 'none','general' or 'phase'

Default: 'none'

The background removal procedure. If 'none', this step is omitted. If 'general' or 'phase', background is removed according to Algorithms 2 and, respectively, 3 in [2]. 

 
- add_dummy

Type: bool

Default: False

Only considered when using WDD for noncircular measurements (ptycho.circular = False).
When False, there are less measurements than WDD needs. Recommendation to change the dimension d to d - window.shape + shift.
When True, dimensional changes are not required. Instead, the algorithm will set the missing diffraction patterns to 0. This results in border in the reconstruction.

- memory_saving

Type: bool

Default: False

If False, the straightforward implementation of WDD constructs banded matrix X = T(xx^*), see Section 3.6.1 in [1]. However, as the number of recovered diagonals gamma is typically much smaller than the dimension d, entries of X are mostly zeros, consuming the memory, especially in 2D. Setting memory_saving to True avoids construction of X and performs magnitude and phase estimation directly from the diagonals. This results in a smaller memory consumption, however, slows down the reconstruction as the phase estimation runs power method instead of using scipy.sparse.linalg.eigsh.    

- xt

Type: np.ndarray

Default: empty

Ground-truth object for testing purposes.

- subspace completing parameters can be ignored
