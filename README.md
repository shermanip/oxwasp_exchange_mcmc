# oxwasp_exchange_mcmc

**Java** implementations of the **Metropolis-Hastings** (Metropolis et al., 1953) (Hastings, 1970), **Adaptive Metropolis-Hastings** (Haario et al., 2001) (Roberts and Rosenthal, 2009), **Hamiltonian Monte Carlo** (Neal, 2011) and **No U-Turn Sampler** (Hoffman and Gelman, 2014).

There exist 2 branches. The [master branch](https://github.com/shermanip/oxwasp_exchange_mcmc) only contains Java implements of the MCMC algorithms and requires the following libraries
* org.ejml [Efficient Java Matrix Library](http://ejml.org/wiki/index.php?title=Main_Page)
* org.apache.commons.math3 [Commons Math: The Apache Commons Mathematics Library](http://commons.apache.org/proper/commons-math/)

The [processing branch](https://github.com/shermanip/oxwasp_exchange_mcmc/tree/processing) is for visualising MCMC in Processing and requires the following libraries
* processing.core.PApplet [Processing](https://processing.org/)
* g4p_controls [G4P (GUI for processing)](http://www.lagers.org.uk/g4p/)

The [ccfe branch](https://github.com/shermanip/oxwasp_exchange_mcmc/tree/ccfe) requires software and data which is only available to staff at the Culham Centre for Fusion Energy.

References
* Haario, H., Saksman, E., Tamminen, J., et al. (2001). An adaptive Metropolis algorithm. _Bernoulli_, 7(2):223-242.
* Hastings, W. K. (1970). Mote Carlo Sampling methods using Markov chains and their applications. _Biometrikia_ 57(1):97-109.
* Hoffman, M. D. and Gelman, A. (2014). The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Mote carlo. _Journal of Machine Learning Research_, 15(1):1593-1623.
* Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., and Teller, E. (1953). Equation of state calculations by fast computing machines. _The Journal of Chemical Physics_, 21(6):1087–1092.
* Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In Brooks, S., Gelman, A., Jones, G., and Meng, X.-L., editors, _Handbook of Markov Chain Monte Carlo_, chapter 5, pages 113–162. CRC press.
* Roberts, G. O. and Rosenthal, J. S. (2009). Examples of adaptive MCMC. _Journal of Computational and Graphical Statistics_, 18(2):349–367.

![alt text](https://github.com/shermanip/oxwasp_exchange_mcmc/blob/master/tex/chain_1.png "Mcmc chain")
![alt text](https://github.com/shermanip/oxwasp_exchange_mcmc/blob/master/tex/processing_rwmh.png "Metropolis-Hastings")
![alt text](https://github.com/shermanip/oxwasp_exchange_mcmc/blob/master/tex/processing_hmc.png "Hamiltonian Monte Carlo")
![alt text](https://github.com/shermanip/oxwasp_exchange_mcmc/blob/master/tex/processing_hmc2.png "Hamiltonian Monte Carlo")
![alt text](https://github.com/shermanip/oxwasp_exchange_mcmc/blob/master/tex/processing_nuts.png "No U-Turn Sampler")
