/*
 *    Copyright 2018-2020 Sherman Lo

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

package uk.ac.warwick.sip.mcmc;

import aliceinnets.python.jyplot.JyPlot;
import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.MatrixFeatures_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.CholeskyDecomposition_F64;
import org.ejml.simple.SimpleMatrix;

/**GLOBAL
 * Example code for MCMC targetting a Normal with random covariance
 * Runs 5 chains
 * Plots trace plot, F statistic, acceptance rate, autocorrelation, autocorrelation of the batches
 * Prints efficiency, log precision, mean and error
 */
public class Global {

  public static void main(String[] args) {

    if (args.length == 0) {
      uk.ac.warwick.sip.mcmcprocessing.Menu.main(args);
    } else {
      String userArg = args[0];
      if (userArg.equals("-bm")) {
        uk.ac.warwick.sip.mcmcprocessing.BrownianMotion.main(args);
      } else if (userArg.equals("-hmc")) {
        uk.ac.warwick.sip.mcmcprocessing.HamiltonianMonteCarlo.main(args);
      } else if (userArg.equals("-nuts")) {
        uk.ac.warwick.sip.mcmcprocessing.NoUTurnSampler.main(args);
      } else if (userArg.equals("-rwmh")) {
        uk.ac.warwick.sip.mcmcprocessing.RandomWalkMetropolisHastings.main(args);
      } else if (userArg.equals("-example")) {
        example();
      } else if (userArg.equals("-test")) {
        Test.main(args);
      }
    }
  }

  public static void example() {
    int nDim = 16;
    int chainLength = 100000;
    MersenneTwister rng = new MersenneTwister(-280845742);
    SimpleMatrix targetCovariance = Global.getRandomCovariance(nDim, rng);
    SimpleMatrix massMatrix = SimpleMatrix.identity(nDim);
    massMatrix = massMatrix.scale(1.0);
    double sizeLeapFrog = 0.5;
    TargetDistribution target = new NormalDistribution(nDim, targetCovariance);

    //for rubin-gelman, need to run additional chains
    int nChain = 5;
    //declare array of chains
    Mcmc [] mcmcArray = new Mcmc [nChain];
    //declare variables for different initial values
    double [] initialValue = new double[target.getNDim()];
    //initial value uses random points from the first chain
    int nIndex;
    //for each chain
    for (int iChain=0; iChain<nChain; iChain++) {
      //instantiate a chain
      //Mcmc chain = new MixtureAdaptiveRwmh(target, chainLength, proposalCovariance, rng);
      //Mcmc chain = new HamiltonianMonteCarlo(target, chainLength, massMatrix, sizeLeapFrog, nLeapFrog, rng) ;
      Mcmc chain = new NoUTurnSampler(target, chainLength,massMatrix, sizeLeapFrog, rng) ;
      //Mcmc chain = new DualAveragingNuts(target, chainLength, massMatrix, nAdaptive, rng) ;
      //for not the first chain, set the initial point using a random point from the first chain
      if (iChain != 0) {
        //use random point from the first chain as the initial value
        nIndex = rng.nextInt(chainLength);
        for (int iDim=0; iDim<target.getNDim(); iDim++) {
          initialValue[iDim] = mcmcArray[0].getChain()[nIndex*target.getNDim()+iDim];
        }
        chain.setInitialValue(initialValue);
      }
      //run the chain and save it
      chain.run();
      mcmcArray[iChain] = chain;
    }
    //plot trace plot for each chain
    for (int i=0; i<nChain; i++) {
      JyPlot tracePlot = new JyPlot();
      tracePlot.figure();
      tracePlot.plot(mcmcArray[i].getChain(0));
      tracePlot.xlabel("number of iterations");
      tracePlot.ylabel("sample");
      tracePlot.show();
      tracePlot.exec();
    }

    //get the gelman rubin statistic
    //plot 2,3,...,nBurnInMax vs F
    int nBurnInMax = 2000;
    GelmanRubinF fStat = new GelmanRubinF(mcmcArray);
    double [] nBurnIn = new double [nBurnInMax-1];
    for (int i=0; i<nBurnIn.length; i++) {
      nBurnIn[i] = (double)(i+2);
    }
    JyPlot fPlot = new JyPlot();
    fPlot.figure();
    fPlot.plot(nBurnIn, fStat.getGelmanRubinFArray(0, nBurnInMax));
    fPlot.xlabel("burn-in");
    fPlot.ylabel("F statistic");
    fPlot.show();
    fPlot.exec();

    //plot acceptance rate
    JyPlot acceptancePlot = new JyPlot();
    acceptancePlot.figure();
    acceptancePlot.plot(mcmcArray[0].getAcceptanceRate());
    acceptancePlot.xlabel("number of iterations");
    acceptancePlot.ylabel("acceptance rate");
    acceptancePlot.show();
    acceptancePlot.exec();


    //plot autocorrelation
    int nLag = 100;
    double [] lag = new double[nLag];
    for (int i=0; i<nLag; i++) {
      lag[i] = (double)(i);
    }
    JyPlot autoCorrelationPlot = new JyPlot();
    autoCorrelationPlot.figure();
    autoCorrelationPlot.stem(lag, mcmcArray[0].getAcf(0, nLag));
    autoCorrelationPlot.hlines(1/Math.sqrt(chainLength), 0, nLag-1, "r");
    autoCorrelationPlot.hlines(-1/Math.sqrt(chainLength), 0, nLag-1, "r");
    autoCorrelationPlot.xlabel("lag");
    autoCorrelationPlot.ylabel("autocorrelation");
    autoCorrelationPlot.show();
    autoCorrelationPlot.exec();

    //plot autocorrelation of the batch
    nLag = 10;
    lag = new double[nLag];
    for (int i=0; i<nLag; i++) {
      lag[i] = (double)(i);
    }
    JyPlot batchAcfPlot = new JyPlot();
    batchAcfPlot.figure();
    batchAcfPlot.stem(lag, mcmcArray[0].getBatchAcf(nLag));
    batchAcfPlot.hlines(1/Math.sqrt(Math.sqrt(chainLength)), 0, nLag-1, "r");
    batchAcfPlot.hlines(-1/Math.sqrt(Math.sqrt(chainLength)), 0, nLag-1, "r");
    batchAcfPlot.xlabel("lag");
    batchAcfPlot.ylabel("autocorrelation");
    batchAcfPlot.show();
    batchAcfPlot.exec();

    //print the efficiency for all chains
    for (int i=0; i<nChain; i++) {
      System.out.println("===Chain "+i+" ===");
      //print the efficiency
      System.out.println("efficiency = "+mcmcArray[i].getEfficiency(0));
      //calculate the posterior statistics using burn in
      mcmcArray[i].calculatePosteriorStatistics(0); //burn in of zero in this example
      //print the monte carlo error
      System.out.println("log precision = "+mcmcArray[i].getDifferenceLnError()[0]);
      //print the mean
      System.out.println("mean = "+mcmcArray[i].getPosteriorExpectation()[0]);
      //print the variance
      SimpleMatrix posteriorCovariance = new SimpleMatrix(mcmcArray[i].getNDim(),
          mcmcArray[i].getNDim(), true, mcmcArray[i].getPosteriorCovariance());
      System.out.println("error = "+Math.sqrt(posteriorCovariance.get(0, 0)));

    }
  }


  /**FUNCTION: CHOLESKY DECOMPOSITION
   * @param x Symmetric matrix to decompose
   * @return cholesky decomposition if possible, other null
   */
  public static SimpleMatrix cholesky(SimpleMatrix x) {
    x = new SimpleMatrix(x);
    if (!MatrixFeatures_DDRM.isSymmetric(x.getDDRM())) {
      return null;
    }
    CholeskyDecomposition_F64<DMatrixRMaj> chol = DecompositionFactory_DDRM.chol(x.numRows(),true);
    if( !chol.decompose(x.getMatrix())) {
      return null;
    }
    return SimpleMatrix.wrap(chol.getT(null));
  }


  /**FUNCTION: GET RANDOM COVARIANCE
   * Generates a random covariance using ZZ' where Z is a dxd matrix of random standard Gaussian
   * @param nDim Number of dimensions
   * @param rng MersenneTwister object to generate random numbers
   * @return SimpleMatrix containing the covariance
   */
  public static SimpleMatrix getRandomCovariance(int nDim, MersenneTwister rng) {
    SimpleMatrix covariance = new SimpleMatrix(nDim, nDim);
    for (int i=0; i<covariance.getNumElements(); i++) {
      covariance.set(i, rng.nextGaussian());
    }
    return covariance.mult(covariance.transpose());
  }

}
