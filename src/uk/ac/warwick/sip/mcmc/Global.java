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
    
    int nDim = 32;
    int chainLength = 10000;
    MersenneTwister rng = new MersenneTwister(-280845742);
    //SimpleMatrix proposalCovariance = SimpleMatrix.identity(nDim).scale(0.01/32.0);
    SimpleMatrix targetCovariance = getRandomCovariance(nDim, rng);
    SimpleMatrix massVector = new SimpleMatrix(nDim,1);
    massVector = massVector.plus(1.0);
    double sizeLeapFrog = 0.5;
    //SimpleMatrix targetCovariance = SimpleMatrix.identity(nDim);
    TargetDistribution target = new NormalDistribution(nDim, targetCovariance);
    
    long timeStart = System.currentTimeMillis();
    RandomWalkMetropolisHastings chain = 
        new NoUTurnSampler(target, chainLength, massVector, sizeLeapFrog,
            rng );
    chain.run();
    System.out.println("Time taken");
    System.out.println(System.currentTimeMillis() - timeStart);
    
    Plot2DPanel tracePlot = new Plot2DPanel();
    DMatrixRMaj trace = chain.chainArray.extractVector(false, 0).getMatrix();
    tracePlot.addLinePlot("trace",trace.data);
    
     // put the PlotPanel in a JFrame, as a JPanel
    JFrame frame = new JFrame("a plot panel");
    frame.setContentPane(tracePlot);
    frame.setSize(800, 600);
    frame.setVisible(true);
    
  }
  
  
  /**FUNCTION: CHOLESKY DECOMPOSITION
   * @param x Symmetric matrix to decompose
   * @return cholesky decomposition if possible, other null
   */
  public static SimpleMatrix cholesky(SimpleMatrix x) {
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
