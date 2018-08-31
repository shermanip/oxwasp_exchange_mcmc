/*
 *    Copyright 2018 Sherman Ip

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

import java.io.PrintWriter;

import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

import uk.ac.warwick.sip.mcmc.NoUTurnSampler.Tree;

public class Test {
  
  static PrintWriter printWriter;
  
  static public void main(String[] args) {
    
    try {
      printWriter = new PrintWriter("test.txt");
      testChain(10, 1000, 3, 1964976895, "Test 1.1");
      testChain(10, 1000, 100, 1964976895, "Test 1.2");
      testChain(10, 1000, 999, 1964976895, "Test 1.3");
      testChain(100, 1000, 999, 2021180607, "Test 1.4");
      testCholesky(20, -1470540617, "Test 2");
      testAcceptStep(10, 100, 1963517091, "Test 3");
      testCopyExtendConstructor(32, 100, 20, -62811111, "Test 4.1");
      testCopyExtendConstructor(32, 1000, 20, -62811111, "Test 4.2");
      testCopyExtendConstructor(32, 1000, 100, -62811111, "Test 4.3");
      testCopyExtendConstructor(64, 1000, 500, -62811111, "Test 4.4");
      testThin(32, 100, 5, 1289694793, "Test 5.1");
      testThin(32, 100, 10, 1289694793, "Test 5.2");
      testThin(32, 1000, 10, 1289694793, "Test 5.3");
      testThin(32, 1000, 50, 1289694793, "Test 5.4");
      testAdaptive(32, 1000, -1538086464, "Test 6.1");
      testAdaptive(32, 1000, -1538086464, "Test 6.2");
      testAdaptive(32, 1000, -1538086464, "Test 6.3");
      testHmc(16, 1742863098, "Test 7.1");
      testHmc(32, 1742863098, "Test 7.2");
      testHmc(64, 1742863098, "Test 7.3");
      testHmc(128, 1742863098, "Test 7.4");
      testTree(32, 1, -1602079425, "Test 8.1");
      testTree(32, 2, 1775435783, "Test 8.2");
      testTree(32, 3, 568478633, "Test 8.3");
      testTree(32, 4, -1728550799, "Test 8.4");
      testTree(32, 5, -951342906, "Test 8.5");
      testTree(32, 6, 201354591, "Test 8.6");
      printWriter.flush();
      printWriter.close();
    } catch (Exception expection) {
      
    }
  }
  
  /**FUNCTION: GET CHAIN
   * Return one implementation of a MCMC
   * @param iChain Integer pointing to which mcmc to instantiate
   * @param nDim Number of dimensions
   * @param chainLength Length of the chain
   * @param rng MersenneTwister to give to the mcmc object
   * @return Instantiate chain
   */
  static public Mcmc getChain(int iChain, int nDim, int chainLength, MersenneTwister rng) {
    //simple Normal target with identity covariance
    SimpleMatrix targetCovariance = SimpleMatrix.identity(nDim);
    TargetDistribution target = new NormalDistribution(nDim, targetCovariance);
    
    //rwmh parameters
    SimpleMatrix proposalCovariance = targetCovariance.scale( Math.pow(2.38,2) / ((double)nDim) );
    
    //hmc parameters
    SimpleMatrix massMatrix = SimpleMatrix.identity(nDim);
    int nLeapFrog = 20;
    double sizeLeapFrog = 0.5;
    int nAdaptive = 100;
      
    //instantiate the chain
    Mcmc chain = null;
    switch (iChain) {
      case 0:
        chain = new RandomWalkMetropolisHastings(target, chainLength, proposalCovariance, rng);
        break;
      case 1:
        chain = new AdaptiveRwmh(target, chainLength, proposalCovariance, rng);
        break;
      case 2:
        chain = new MixtureAdaptiveRwmh(target, chainLength, proposalCovariance, rng);
        break;
      case 3:
        chain = new HamiltonianMonteCarlo(target, chainLength, massMatrix, sizeLeapFrog,
            nLeapFrog, rng);
        break;
      case 4:
        chain = new NoUTurnSampler(target, chainLength, massMatrix, sizeLeapFrog, rng);
        break;
      case 5:
        chain = new DualAveragingNuts(target, chainLength, massMatrix, nAdaptive, rng);
        break;
      default:
        break;
    }
    return chain;
  }
  
  /**FUNCTION: COPY CONSTRUCTOR
   * Calls the chain's copy and extend constructor
   * @param iChain Pointer to the class of the object
   * @param chain The chain to call the copy and extend constructor
   * @param nMoreSteps Number of steps to extend
   * @return The chain copied and extended
   */
  static Mcmc copyConstructor(int iChain, Mcmc chain, int nMoreSteps) {
    //call the copy and extend constructor
    switch (iChain) {
      case 0:
        chain = new RandomWalkMetropolisHastings((RandomWalkMetropolisHastings)chain, nMoreSteps);
        break;
      case 1:
        chain = new AdaptiveRwmh((AdaptiveRwmh)chain, nMoreSteps);
        break;
      case 2:
        chain = new MixtureAdaptiveRwmh((MixtureAdaptiveRwmh)chain, nMoreSteps);
        break;
      case 3:
        chain = new HamiltonianMonteCarlo((HamiltonianMonteCarlo)chain, nMoreSteps);
        break;
      case 4:
        chain = new NoUTurnSampler((NoUTurnSampler)chain, nMoreSteps);
        break;
      case 5:
        chain = new DualAveragingNuts((DualAveragingNuts)chain, nMoreSteps);
        break;
      default:
        break;
    }
    return chain;
  }
  
  /**FUNCTION: TEST CHAIN
   * Test the member variables chainMean and chainCovariance, these are recursive estimators.
   * The recursive estimators are comapred with the non-recursive estimators.
   * The non-recursive estimators uses the data after the chain finished running.
   * The stanard Normal distribution is sampled using the optimial proposal.
   * Test if nStep increments at every step
   * @param nDim Number of dimensions the target has
   * @param chainLength The length of the chain
   * @param nStep Number of steps to take in the chain,
   *     the test will be done when the steps are taken
   * @param seed Random seed for the MersenneTwister
   * @param name Name of the test, this will be printed
   */
  static void testChain(int nDim, int chainLength, int nStep,
      int seed, String name) {
    
    printWriter.println("==========");
    printWriter.println(name);
    
    //for each mcmc class
    for (int iMcmc=0; iMcmc<6; iMcmc++) {
      
      boolean isNStep = true; //tests if nStep increments
      boolean isNSample = true; //tests if nSample increments
      boolean isAllocation = true; //tests if the sample has been allocated to chainArray correctly
      
      //random number generator
      MersenneTwister rng = new MersenneTwister(seed);
      
      Mcmc chain = getChain(iMcmc, nDim, chainLength, rng);
      
      //set random initial value
      double [] initial = getRandomVector(nDim, rng).getDDRM().getData();
      chain.setInitialValue(initial);
      
      //instantiate column vector for the current value of the chain
      SimpleMatrix x = chain.chainArray.extractVector(true, 0);
      CommonOps_DDRM.transpose(x.getDDRM());
      
      //run the chain for nStep
      for (int i=0; i<nStep; i++) {
        
        //copy x to for allocation test
        SimpleMatrix xBefore = new SimpleMatrix(x);
        
        //do mcmc step
        chain.step(x);
        //save x to the chain array
        chain.setCurrentStep(x);
        
        //check if chain.nStep has been incremented
        if (chain.nStep != (i+1)) {
          isNStep = false;
        }
        //check if chain nSample increments
        if (chain.nSample != (i+2)) {
          isNSample = false;
        }
        //check if x has been copied to chainArray
        //also check if the next sample in chainArray are all zeros
        //also check if the sample before is left unchanged
        for (int iDim=0; iDim<chain.getNDim(); iDim++) {
          if (chain.chainArray.get(i+1, iDim)!=x.get(iDim)) {
            isAllocation = false;
          }
          if (i!=(nStep-1)) {
            if (chain.chainArray.get(i+2, iDim)!=0) {
              isAllocation = false;
            }
          }
          if (i!=0) {
            if (chain.chainArray.get(i, iDim)!=xBefore.get(iDim)) {
              isAllocation = false;
            }
          }
          
        }
        
      }
      
      //declare array for storing squared error
      //entry one for the mean
      //entry two for the covariance
      double [] squaredErrorArray = new double [2];
      
      //get the mcmc samples after nStep
      SimpleMatrix designMatrix = chain.chainArray.extractMatrix(0, nStep+1, 0, nDim);
      
      //work out the sample mean and get and save the squared error
      SimpleMatrix sampleMean = getSampleMean(designMatrix);
      SimpleMatrix squaredError = (chain.chainMean.minus(sampleMean)).elementPower(2);
      squaredErrorArray[0] = squaredError.elementSum();
      
      //work out the sample covariance, get and save the squared error
      SimpleMatrix sampleCovariance = getSampleCovariance(designMatrix, sampleMean);
      squaredError = (chain.chainCovariance.minus(sampleCovariance)).elementPower(2);
      squaredErrorArray[1] = squaredError.elementSum();
      
      //print the squared error and return it
      printWriter.println(chain.getClass().getName());
      printWriter.println("Squared error in mean = "+squaredErrorArray[0]);
      printWriter.println("Squared error in covariance = "+squaredErrorArray[1]);
      
      printWriter.println("pass nStep test = "+isNStep);
      printWriter.println("pass nSample test = "+isNSample);
      printWriter.println("pass allocation test = "+isAllocation);
    }
  }
  
  /**FUNCTION: GET SAMPLE MEAN
   * Returns the unbiased estimator of the mean
   * @param designMatrix Collection of data, rows represent each data
   *     columns represent each dimension
   * @return Unbiased estimator of the sample mean (column vector)
   */
  static public SimpleMatrix getSampleMean(SimpleMatrix designMatrix) {
    int n = designMatrix.numRows(); //number of datapoints
    int p = designMatrix.numCols(); //number of dimensions
    //instantiate column vector for the sample mean
    SimpleMatrix sampleMean = new SimpleMatrix(p, 1);
    //for each dimension, work out the sample mean
    //combine all the dimensions together and return it
    for (int i=0; i<p; i++) {
      sampleMean.set(i, designMatrix.extractVector(false, i).elementSum());
    }
    return sampleMean.divide((double)n);
  }
  
  /**FUNCTION: GET SAMPLE COVARIANCE
   * Returns the unbiased estimator of the covariance
   * @param designMatrix Collection of data, rows represent each data,
   *     columns represent each dimension
   * @param sampleMean The mean estimator in a column vector format
   * @return Unbiased estimator of the covariance
   */
  static public SimpleMatrix getSampleCovariance(SimpleMatrix designMatrix,
      SimpleMatrix sampleMean) {
    int n = designMatrix.numRows(); //number of datapoints
    int p = designMatrix.numCols(); //number of dimensions
    //instantiate a matrix for the sample covariance
    SimpleMatrix sampleCovariance = new SimpleMatrix(p,p);
    //for each data point, update the sample covariance
    //this is the sum of squared difference from the mean
    for (int i=0; i<n; i++) {
      SimpleMatrix r = (designMatrix.extractVector(true, i).transpose()).minus(sampleMean);
      sampleCovariance = sampleCovariance.plus(r.mult(r.transpose()));
    }
    //normalise the sum of squared difference using n-1
    //this is the unbiased estimator
    return sampleCovariance.divide((double)(n - 1));
  }
  
  /**FUNCTION: TEST CHOLESKY
   * Check if Global.cholesky() does not modify the parameter
   * Check if Global.cholesky() returns null
   * Check if 
   * @param nDim Number of dimension
   * @param seed Seed for Mersenne Twister
   * @param name Name of the test
   * @return true if pass the test
   */
  static void testCholesky(int nDim, int seed, String name) {
    printWriter.println("==========");
    printWriter.println(name);
    int nTest = 10; // number of tests
    MersenneTwister rng = new MersenneTwister(seed);
    boolean isNoNull = true;
    boolean isNull = true;
    boolean isNotModify = true;
    //for n tests
    for (int iTest=0; iTest<nTest; iTest++) {
      //get a random covariance and clone it
      SimpleMatrix cov = Global.getRandomCovariance(nDim, rng);
      SimpleMatrix covClone = new SimpleMatrix(cov);
      SimpleMatrix chol = Global.cholesky(cov); //do cholesky decomposition
      //check if chol is null
      if (chol==null) {
        isNoNull = false;
      }
      //check if the difference of cov before and after chol is 0
      if (!cov.isIdentical(covClone, 0)) {
        isNotModify = false;
      };
      //test is L*LT is similar
      SimpleMatrix cholCholT = chol.mult(chol.transpose());
      double sumDiffSquared = cholCholT.minus(cov).elementPower(2).elementSum();
      printWriter.println("Squared error in Cholesky decomposition = "+sumDiffSquared);
      
      //check if cholesky of chol of N(0,1) outputs null
      SimpleMatrix randGaussian = new SimpleMatrix(nDim, nDim);
      for (int i=0; i<randGaussian.getNumElements(); i++) {
        randGaussian.set(i, rng.nextGaussian());
      }
      chol = Global.cholesky(randGaussian); //do cholesky decomposition
      if (chol!=null) {
        isNull = false;
      }
    }
    printWriter.println("pass no null test = "+isNoNull);
    printWriter.println("pass null test = "+isNull);
    printWriter.println("pass modify test = "+isNotModify);
    
  }
  
  /**FUNCTION: GET RANDOM VECTOR
   * @param nDim
   * @param rng
   * @return Column vector, random Gaussian
   */
  static SimpleMatrix getRandomVector(int nDim, MersenneTwister rng) {
    SimpleMatrix x = new SimpleMatrix(nDim, 1);
    for (int iDim=0; iDim<nDim; iDim++) {
      x.set(iDim, rng.nextGaussian());
    }
    return x;
  }
  
  /**FUNCTION: TEST ACCEPT STEP
   * Checks if the method acceptStep modifies its parameters accordingly
   * Checks if nAccept is incremented correctly
   * @param nDim
   * @param chainLength
   * @param seed
   * @param name
   */
  static void testAcceptStep(int nDim, int chainLength, int seed, String name) {
    printWriter.println("==========");
    printWriter.println(name);
    MersenneTwister rng = new MersenneTwister(seed);
    int nTest = 100; //number of times to repeat the test
    boolean isModifyTest = true;
    boolean isNAcceptTest = true;
    
    //for the rwmh family of mcmc
    for (int iChain=0; iChain<3; iChain++) {
      
      Mcmc chain = null;
      for (int i=0; i<nTest; i++) {
        
        chain = getChain(iChain, nDim, chainLength, rng);
        
        //check with accept probability 0
        SimpleMatrix position = getRandomVector(nDim, rng);
        SimpleMatrix proposal = getRandomVector(nDim, rng);
        SimpleMatrix positionBefore = new SimpleMatrix(position);
        SimpleMatrix proposalBefore = new SimpleMatrix(proposal);
        chain.acceptStep(0.0, position, proposal);
        //check if both parameters are not modified
        if (!positionBefore.isIdentical(position,0)) {
          isModifyTest = false;
        }
        if (!proposalBefore.isIdentical(proposal,0)) {
          isModifyTest = false;
        }
        //test if nAccept = 0
        if (chain.nAccept != 0) {
          isNAcceptTest = false;
        }
        
        //check with accept probability 1.0
        position = getRandomVector(nDim, rng);
        proposal = getRandomVector(nDim, rng);
        positionBefore = new SimpleMatrix(position);
        proposalBefore = new SimpleMatrix(proposal);
        chain.acceptStep(1.0, position, proposal);
        //check if position is modified and proposal is not modified
        if (!proposal.isIdentical(position,0)) {
          isModifyTest = false;
        }
        if (!proposalBefore.isIdentical(proposal,0)) {
          isModifyTest = false;
        }
        //test if nAccept = 0
        if (chain.nAccept != 1) {
          isNAcceptTest = false;
        }
        
        //check with accept probability 1.0
        position = getRandomVector(nDim, rng);
        proposal = getRandomVector(nDim, rng);
        positionBefore = new SimpleMatrix(position);
        proposalBefore = new SimpleMatrix(proposal);
        chain.acceptStep(0.5, position, proposal);
        //check if the proposal is not modified
        if (!proposalBefore.isIdentical(proposal, 0)) {
          isModifyTest = false;
        }
        //check if the position has been modified with an accept step
        if (chain.nAccept == 2) {
          if (!proposal.isIdentical(position, 0)) {
            isModifyTest = false;
          }
        //check if the position has not been modified with a reject step
        } else {
          if (!positionBefore.isIdentical(position, 0)) {
            isModifyTest = false;
          }
        }
      }
      
      printWriter.println(chain.getClass().getName());
      printWriter.println("pass modify test = "+isModifyTest);
      printWriter.println("pass nAccept test = "+isNAcceptTest);
      
    }
    
  }
  
  /**FUNCTION: TEST COPY AND EXTEND CONSTRUCTOR
   * Test the copy and extend constructor, runs 2 chains of chainLength, 1 stops at subChainLength
   * and then calls the copy and extend constructor to run the remaining steps
   * Checks the 2 chains has the same nStep
   * Checks the 2 chains has the same nSample
   * Checks the 2 chains has the exact same chainArray
   * @param nDim Number of dimensions
   * @param chainLength Length of chain
   * @param subChainLength Length of the first run
   * @param seed for rng
   * @param name
   */
  static void testCopyExtendConstructor(int nDim, int chainLength, int subChainLength,
      int seed, String name) {
    
    printWriter.println("==========");
    printWriter.println(name);
    
    //for the rwmh family of mcmc
    for (int iChain=0; iChain<6; iChain++) {
      
      //boolean for the tests
      boolean isNStep = true;
      boolean isNSample = true;
      boolean isSame = true;
      
      //get the chain and run it all the way
      MersenneTwister rng = new MersenneTwister(seed);
      Mcmc chain = getChain(iChain, nDim, chainLength, rng);
      chain.run();
      
      //get the chain, run it for subChainLength, copy and extend, then run the remaining steps
      rng = new MersenneTwister(seed);
      Mcmc chainUseCopyConstructor = getChain(iChain, nDim, subChainLength, rng);
      chainUseCopyConstructor.run();
      chainUseCopyConstructor = copyConstructor(iChain, chainUseCopyConstructor,
          chainLength - subChainLength);
      chainUseCopyConstructor.run();
      
      //check if nStep are the same
      if (chain.nStep != chainUseCopyConstructor.nStep) {
        isNStep = false;
      }
      //check if nSample are the same
      if (chain.nSample != chainUseCopyConstructor.nSample) {
        isNSample = false;
      }
      //check if chainArray are the same
      if (!chain.chainArray.isIdentical(chainUseCopyConstructor.chainArray,0)) {
        isSame = false;
      };
      
      printWriter.println(chain.getClass().getName());
      printWriter.println("pass nStep test = "+isNStep);
      printWriter.println("pass nSample test = "+isNSample);
      printWriter.println("pass isSame test = "+isSame);
      
    }
  }
  
  /**FUNCTION: TEST THIN
   * Test how the chain behaves with thinning
   * Checks if the number of rows in chainArray is correct
   * Checks if nStep is correct, this is the number of MCMC steps
   * Checks if nSample is correct
   * @param nDim
   * @param chainLength
   * @param nThin
   * @param seed
   * @param name
   */
  static void testThin(int nDim, int chainLength, int nThin, int seed, String name) {
    printWriter.println("==========");
    printWriter.println(name);
    
    //for the rwmh family of mcmc
    for (int iChain=0; iChain<6; iChain++) {
      //boolean for the tests
      boolean isArrayHeight = true;
      boolean isNStep = true;
      boolean isNSample = true;
      
      //get the chain and run it all the way
      MersenneTwister rng = new MersenneTwister(seed);
      Mcmc chain = getChain(iChain, nDim, chainLength, rng);
      chain.setNThin(nThin);
      chain.run();
      
      if (chain.chainArray.numRows() != chainLength) {
        isArrayHeight = false;
      }
      if (chain.nStep != ( (chainLength-1)*nThin ) ) {
        isNStep = false;
      }
      if (chain.nSample != (chainLength)) {
        isNSample = false;
      }
      printWriter.println(chain.getClass().getName());
      printWriter.println("pass chainArray height test = "+isArrayHeight);
      printWriter.println("pass nStep test = "+isNStep);
      printWriter.println("pass nSample test = "+isNSample);
    }
  }
  
  /**FUNCTION: TEST ADAPTIVE
   * Tests for AdaptiveRwmh and MixtureAdaptiveRwmh
   * Tests if the chain only adapts after nStepTillAdaptive steps
   * Test if the safety proposal is used when adapting (only for MixtureAdaptiveRwmh)
   * Test if the safety is different from the initial
   * @param nDim Number of dimensions
   * @param chainLength Length of chain
   * @param seed Seed for rng
   * @param name Name of the test
   */
  static void testAdaptive(int nDim, int chainLength, int seed, String name) {
    
    //print test name
    printWriter.println("==========");
    printWriter.println(name);
    
    //for the rwmh family of mcmc
    for (int iChain=1; iChain<=2; iChain++) {
      
      //boolean for the tests
      boolean isUseInitialProposal = true; //if the initial proposal is used in non-adaptive stage
      //if uses proposal which is different from the intial in adaptive stage
      boolean isAdapting = false;
      //if uses the initial proposal in the adaptive stage
      boolean isSaftey = false;
      
      //instantiate the chain
      MersenneTwister rng = new MersenneTwister(seed);
      AdaptiveRwmh chain = (AdaptiveRwmh) getChain(iChain, nDim, chainLength, rng);
      //copy the proposal covariance
      SimpleMatrix proposalCovarianceChol = new SimpleMatrix(chain.proposalCovarianceChol);
      
      //instantiate column vector for the current value of the chain
      SimpleMatrix x = chain.chainArray.extractVector(true, 0);
      CommonOps_DDRM.transpose(x.getDDRM());
      
      //run the chain for nStep
      for (int i=0; i<(chainLength-1); i++) {
        
        //do mcmc step
        chain.step(x);
        //save x to the chain array
        chain.setCurrentStep(x);
        
        //test in the non-adaptive stage, the chain uses the initial proposal
        if (i < chain.nStepTillAdaptive) {
          if(!proposalCovarianceChol.isIdentical(chain.proposalCovarianceChol, 0.0)) {
            isUseInitialProposal = false;
          }
        } else { //else this is the adaptive stage
          //check if the proposal is not the same as the initial
          if(!proposalCovarianceChol.isIdentical(chain.proposalCovarianceChol, 0.0)) {
            isAdapting = true;
          }
          //check if the proposal is the same as the initial
          if(proposalCovarianceChol.isIdentical(chain.proposalCovarianceChol, 0.0)) {
            isSaftey = true;
          }
        }
      }
      //print results of the test
      printWriter.println(chain.getClass().getName());
      printWriter.println("pass useInitialProposal test = "+isUseInitialProposal);
      printWriter.println("pass adapting test = "+isAdapting);
      //only MixtureAdaptiveRwmh uses the saftey proposal, print the result of it
      if (iChain==2) {
        printWriter.println("pass saftey test = "+isSaftey);
      }
    }
  }
  
  
  /**FUNCTION: TEST HMC
   * Test if the method getHamiltonian doesn't change the parameters
   * Test if the positionStep and momentumStep methods change the correct parameters
   * Test if momentumStep steps produce the same result when doing one leap frog vs 2 half leap frog
   * @param nDim Number of dimensions
   * @param seed Seed for the rng
   * @param name Name of the test
   */
  static void testHmc(int nDim, int seed, String name) {
    
    //print name of the test
    printWriter.println("==========");
    printWriter.println(name);
    
    //for each mcmc class
    for (int iMcmc=3; iMcmc<6; iMcmc++) {
      
      //booleans for the tests
      //test if the hamiltonian change the parameters
      boolean isHamiltonianModifyParameterTest = true;
      //test if the momentum step modify the parameters correctly
      boolean isMomentumStepModifyTest = true;
      //test if the position step modify the parameters correctly
      boolean isPositionStepModifyTest = true;
      
      //random number generator and chain
      MersenneTwister rng = new MersenneTwister(seed);
      HamiltonianMonteCarlo chain = (HamiltonianMonteCarlo) getChain(iMcmc, nDim, 100, rng);
      
      //instantiate column vector for the current value of the chain
      SimpleMatrix x = chain.chainArray.extractVector(true, 0);
      CommonOps_DDRM.transpose(x.getDDRM());
      
      //instantiate random position and random momentum
      SimpleMatrix position = getRandomVector(nDim, rng);
      SimpleMatrix momentum = chain.getMomentum();
      
      //copy position and momentum, this is then compared after calling getHamiltonian
      SimpleMatrix positionCopy = new SimpleMatrix(position);
      SimpleMatrix momentumCopy = new SimpleMatrix(momentum);
      chain.getHamiltonian(position, momentum);
      //check if the copy of position and momentum are the same
      if ( (!position.isIdentical(positionCopy, 0)) || (!momentum.isIdentical(momentumCopy, 0))) {
        isHamiltonianModifyParameterTest = false;
      }
      
      //copy the position and momentum vectors
      //test if the momentumStep modify the momentum vector only
      positionCopy = new SimpleMatrix(position);
      momentumCopy = new SimpleMatrix(momentum);
      chain.momentumStep(position, momentum, true);
      if ( !position.isIdentical(positionCopy, 0) ) {
        isMomentumStepModifyTest = false;
      }
      if ( momentum.isIdentical(momentumCopy, 0) ) {
        isMomentumStepModifyTest = false;
      }
      
      //copy the position and momentum vectors
      //test if the positionStep modify the position vector only
      positionCopy = new SimpleMatrix(position);
      momentumCopy = new SimpleMatrix(momentum);
      chain.positionStep(position, momentum);
      if ( position.isIdentical(positionCopy, 0) ) {
        isPositionStepModifyTest = false;
      }
      if ( !momentum.isIdentical(momentumCopy, 0) ) {
        isPositionStepModifyTest = false;
      }
      
      //make two copies of the momentum vector
      //one copy takes a full step
      //the other copy take 2 half steps
      //compare the two, they should be similar, look at the squared error
      SimpleMatrix momentumHalfSteps = new SimpleMatrix(momentum);
      SimpleMatrix momentumFullStep = new SimpleMatrix(momentum);
      chain.momentumStep(position, momentumHalfSteps, true);
      chain.momentumStep(position, momentumHalfSteps, true);
      chain.momentumStep(position, momentumFullStep, false);
      double squaredError = momentumFullStep.minus(momentumHalfSteps).elementPower(2).elementSum();
      
      //print results of the test
      printWriter.println(chain.getClass().getName());
      printWriter.println("pass hamiltonian modification test = "+isHamiltonianModifyParameterTest);
      printWriter.println("pass momentum step modification test = "+isMomentumStepModifyTest);
      printWriter.println("pass position step modification test = "+isPositionStepModifyTest);
      printWriter.println("squared error between one leap frog and 2 half leap frog = "
          +squaredError);
    }
  }
  
  /**FUNCTION: TEST TREE
   * Test if the parameters of constructing new trees are left unmodified
   * Test if the height of tree instantiated from a seed, growing and building is correct
   * Test if the subTree is of the correct height
   * Test if growing the tree leaves the correct parameters unmodified
   * @param nDim
   * @param nGrow
   * @param seed
   * @param name
   */
  static void testTree(int nDim, int nGrow, int seed, String name) {
    
    //print name of the test
    printWriter.println("==========");
    printWriter.println(name);
    
    //for each mcmc class
    for (int iMcmc=4; iMcmc<6; iMcmc++) {
      
      boolean isHeightZero = true;
      boolean isHeightCorrectFromSeed = true;
      boolean isSubTreeHeightTest= true;
      boolean isModifyFromSeedTest = true;
      boolean isGrowBackTest = true;
      boolean isGrowForwardTest = true;
      
      //random number generator and chain
      MersenneTwister rng = new MersenneTwister(seed);
      NoUTurnSampler chain = (NoUTurnSampler) getChain(iMcmc, nDim, 100, rng);
      
      //instantiate random position and random momentum
      SimpleMatrix position = getRandomVector(nDim, rng);
      SimpleMatrix momentum = chain.getMomentum();
      chain.sampleSliceVariable(chain.getHamiltonian(position, momentum));
      
      //make a copy of the position and momentum before planting and growing the tree
      SimpleMatrix positionCopy = new SimpleMatrix(position);
      SimpleMatrix momentumCopy = new SimpleMatrix(momentum);
      
      
      //instantiate a tree of height 0, check the height
      Tree baseTree = chain.newTree(position, momentum);
      if (baseTree.height != 0) {
        isHeightZero = false;
      }
      //count the number of times the grow() method has been called
      int growCounter = 0;
      
      //for nGrow times
      for (int iStep=0; iStep<nGrow; iStep++) {
        
        //if no u turn has been made
        if (baseTree.hasNoUTurn) {
          //grow the tree
          boolean direction = rng.nextBoolean();
          baseTree.grow(direction);
          baseTree.bloom();
          growCounter++;
          
          //if going forward in time, check the backward position vectors are the same
          if (direction) {
            if (!baseTree.subTree.positionBackward.isIdentical(baseTree.positionBackward, 0)) {
              isGrowForwardTest = false;
            }
          //else going back in time, check the forward position vectors are the same
          } else {
            if (!baseTree.subTree.positionForward.isIdentical(baseTree.positionForward, 0)) {
              isGrowBackTest = false;
            }
          }
        }
      }
      
      //check if the tree height corresponds to the number of times grow() was called
      if (baseTree.height != growCounter) {
        isHeightCorrectFromSeed = false;
      }
      //check if the sub height corresponds to the number of times grow() was called - 1
      if (baseTree.subTree.height != (growCounter - 1)) {
        isSubTreeHeightTest = false;
      }
      //check if the position and momentum vectors are left unmodified
      if ( (!position.isIdentical(positionCopy, 0)) || (!momentum.isIdentical(momentumCopy, 0)) ) {
        isModifyFromSeedTest = false;
      }
      
      //tests for calling the constructor for instantiating a tree of a given height
      boolean isModifyFromConstructorTest = true;
      boolean isHeightCorrectFromConstructor = true;
      //make a copy of the position and momentum
      positionCopy = new SimpleMatrix(position);
      momentumCopy = new SimpleMatrix(momentum);
      
      //instantiate a tree of nGrow height
      NoUTurnSampler.Tree tree = chain.newTree(position, momentum, rng.nextBoolean(), nGrow);
      //check if the parameters are left unmodified
      if ( (!position.isIdentical(positionCopy, 0)) || (!momentum.isIdentical(momentumCopy, 0)) ) {
        isModifyFromConstructorTest = false;
      }
      
      //if the tree has no u turn, check if the height corresponds to nGrow
      if (tree.hasNoUTurn) {
        if (tree.height != nGrow) {
          isHeightCorrectFromConstructor = false;
        }
      //else the tree has made a u turn, the height of the tree should be less than nGrow
      } else {
        if (tree.height > nGrow) {
          isHeightCorrectFromConstructor = false;
        }
      }
      
      //print results of the test
      printWriter.println(chain.getClass().getName());
      printWriter.println("pass tree height zero test = "+isHeightZero);
      printWriter.println("pass tree height from seed test = "+isHeightCorrectFromSeed);
      printWriter.println("pass subTree height from seed test = "+isSubTreeHeightTest);
      printWriter.println("pass tree modify from seed test = "+isModifyFromSeedTest);
      printWriter.println("pass grow forward from seed test = "+isGrowForwardTest);
      printWriter.println("pass grow backward from seed test = "+isGrowBackTest);
      printWriter.println("pass tree modify from constructor test = "+isModifyFromConstructorTest);
      printWriter.println("pass tree height from constructor test = "
          +isHeightCorrectFromConstructor);
    }
  }
  
}
