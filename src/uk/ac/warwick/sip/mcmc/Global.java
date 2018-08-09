package uk.ac.warwick.sip.mcmc;

import javax.swing.JFrame;
import org.apache.commons.math3.random.MersenneTwister;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.CholeskyDecomposition_F64;
import org.ejml.simple.SimpleMatrix;
import org.math.plot.Plot2DPanel;

public class Global {
  
  public static void main(String[] args) {
    int nDim = 16;
    int chainLength = 10000;
    MersenneTwister rng = new MersenneTwister(-280845742);
    SimpleMatrix targetCovariance = SimpleMatrix.identity(nDim);
    SimpleMatrix proposalCovariance = targetCovariance.scale(Math.pow(0.001, 2)/((double)nDim));
    SimpleMatrix massMatrix = SimpleMatrix.identity(nDim);
    massMatrix = massMatrix.scale(1.0);
    int nLeapFrog = 100;
    double sizeLeapFrog = 0.5;
    TargetDistribution target = new NormalDistribution(nDim, targetCovariance);
    int nAdaptive = 100;
    
    int nChain = 1;
    Mcmc [] chainArray = new Mcmc [nChain];
    SimpleMatrix initialPositionScale = SimpleMatrix.identity(nDim);
    initialPositionScale = initialPositionScale.scale(0.0);
    
    for (int iChain=0; iChain<nChain; iChain++) {
      //chainArray[iChain] =  new MixtureAdaptiveRwmh(target, chainLength, proposalCovariance, rng) ;
      chainArray[iChain] = new HamiltonianMonteCarlo(target, chainLength,massMatrix, sizeLeapFrog, nLeapFrog, rng) ;
      //chainArray[iChain] = new NoUTurnSampler(target, chainLength,massMatrix, sizeLeapFrog, rng) ;
      //chainArray[iChain] = new DualAveragingNuts(target, chainLength, massMatrix, nAdaptive, rng) ;
      //chainArray[iChain].setNThin(50);
      SimpleMatrix initial = new SimpleMatrix(nDim, 1);
      for (int i=0; i<nDim; i++) {
        initial.set(i, rng.nextGaussian());
      }
      initial = initialPositionScale.mult(initial);
      chainArray[iChain].setInitialValue(initial.getDDRM().getData());
      chainArray[iChain].run();
    }
    //chainArray[0] =  new HomogeneousRwmh((HomogeneousRwmh)chainArray[0], 759);
    //chainArray[0].run();
    
    for (int iChain=0; iChain<nChain; iChain++) {
      Plot2DPanel tracePlot = new Plot2DPanel();
      double [] trace = chainArray[iChain].getChain(0);
      tracePlot.addLinePlot("trace",trace);
      
       // put the PlotPanel in a JFrame, as a JPanel
      JFrame frame = new JFrame("a plot panel");
      frame.setContentPane(tracePlot);
      frame.setSize(800, 600);
      frame.setVisible(true);
      
      Plot2DPanel acceptPlot = new Plot2DPanel();
      double [] acceptArray = chainArray[iChain].getAcceptanceRate();
      acceptPlot.addLinePlot("accept",acceptArray);
      
       // put the PlotPanel in a JFrame, as a JPanel
      JFrame acceptFrame = new JFrame("a plot panel");
      acceptFrame.setContentPane(acceptPlot);
      acceptFrame.setSize(800, 600);
      acceptFrame.setVisible(true);
    }
    
    
  }
  
  
  /**FUNCTION: CHOLESKY DECOMPOSITION
   * @param x Symmetric matrix to decompose
   * @return cholesky decomposition if possible, other null
   */
  public static SimpleMatrix cholesky(SimpleMatrix x) {
    x = new SimpleMatrix(x);
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
