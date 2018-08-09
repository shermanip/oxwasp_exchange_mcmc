package jet.apps.lappel.voEqui.magnetics;

import static seed.matrix.Mat.*;

import algorithmrepository.Algorithms;
import algorithmrepository.Interpolation1D;
import algorithmrepository.LinearInterpolation1D;
import jet.libes.apps.LiBesDataSourcePreanalysed;
import jet.minerva.interferometry.InterferometerDataSourceJET;
import jet.minerva.reflectometry.JetKG10DataSourcePreAnalysed;
import jet.ts.apps.KE11DataSource;
import minerva.general.MinervaSettingsProperty;
import oneLiners.OneLiners;
import seed.deepend.compilers.AccessorCompiler;
import seed.deepend.function.Constant;
import seed.deepend.function.ConstantArray;
import seed.deepend.function.ConstantInt;
import seed.deepend.function.CubicInterpolation2DNode;
import seed.deepend.function.EvalScalarFunctionND;
import seed.deepend.function.FillArray;
import seed.deepend.function.FillBooleanArray;
import seed.deepend.function.LinSpaceArray;
import seed.deepend.function.binaryoperators.Add;
import seed.deepend.function.binaryoperators.Divide;
import seed.deepend.function.binaryoperators.Mul;
import seed.deepend.function.binaryoperators.Pow;
import seed.deepend.function.binaryoperators.Subtract;
import seed.deepend.function.unaryoperators.Mean;
import seed.minerva.Differentation1D;
import seed.minerva.GraphicalModel;
import seed.minerva.Multivariate;
import seed.minerva.MultivariateNormal;
import seed.minerva.Normal;
import seed.minerva.ProbabilityNode;
import seed.minerva.TruncatedDistribution;
import seed.minerva.TruncatedMultivariateNormal;
import seed.minerva.Uniform;
import seed.minerva.deepend.MinervaSettings;
import seed.minerva.deepend.Source;
import seed.minerva.diagnostics.ChannelDataEnable;
import seed.minerva.diagnostics.FirstWallJET;
import seed.minerva.diagnostics.LineOfSightSystem;
import seed.minerva.diagnostics.ServiceManager;
import seed.minerva.diagnostics.UnitManager;
import seed.minerva.diagnostics.magnetics.FluxLoopsDataSourceJET2;
import seed.minerva.diagnostics.magnetics.Fluxloops2D;
import seed.minerva.diagnostics.magnetics.JETMagneticDiagnosticsDataSource;
import seed.minerva.diagnostics.magnetics.PickupCoils2D;
import seed.minerva.diagnostics.magnetics.PickupCoils2DDataSourceJET2;
import seed.minerva.diagnostics.magnetics.SaddleCoils;
import seed.minerva.diagnostics.magnetics.SaddleCoilsDataSourceJET2;
import seed.minerva.diagnostics.mse.MSE;
import seed.minerva.diagnostics.mse.MSEDataSourceJET;
import seed.minerva.diagnostics.ts.jet.HRTSDataSourcePreanalysed;
import seed.minerva.equilibrium.EquilibriumModelOnBeamSet;
import seed.minerva.magnetics.MagneticConfigurationSource;
import seed.minerva.magnetics.MagneticModelAxiSym2;
import seed.minerva.magnetics.ToroidalCurrentBeamCalcs;
import seed.minerva.magnetics.jet.DensityToBeamsAdapter;
import seed.minerva.magnetics.jet.FluxContouringOps;
import seed.minerva.magnetics.jet.JETCoilCurrentsDataSource;
import seed.minerva.magnetics.jet.JetEfitPpfPsiDataSource;
import seed.minerva.magnetics.jet.JetMagneticConfigurationDataSource;
import seed.minerva.magnetics.jet.PoloidalFluxGrid;
import seed.minerva.magnetics.jet.PulseInfo;
import seed.minerva.magnetics.jet.QProfile;
import seed.minerva.nodetypes.GPHigdonSwallKernGaussianKernel1D;
import seed.minerva.nodetypes.GPSquaredExponentialND;
import seed.minerva.nodetypes.Interpolation1DNode;
import seed.minerva.physics.Const;
import seed.minerva.physics.FluxCoordinateTransform;
import seed.minerva.physics.FluxMapWithPrivateRegion;
import seed.minerva.physics.LineOfSightIntegration;
import seed.minerva.physics.equilibrium.CurrentAndEnergyMoments;
import seed.minerva.physics.equilibrium.PoloidalCurrentFluxFromDifferential;
import seed.minerva.physics.equilibrium.PressureFromDifferential;
import seed.minerva.polarimetry.PolarimeterPredictionNode;
import seed.minerva.polarimetry.PolarimetryDataSourceJET;
import seed.minerva.toBeGeneral.BoundedCellGrid2D;
import seed.minerva.toBeGeneral.EquiProfileKnotSpacing;
import seed.minerva.toBeGeneral.EvalScalar2DOnGrid;
import seed.minerva.toBeGeneral.EvalScalarFunction1D;
import seed.minerva.toBeGeneral.ExtractEnable;
import seed.minerva.toBeGeneral.FixedKnot1DProfileConfig;
import seed.minerva.toBeGeneral.FluxMappedFunction;
import seed.minerva.ts.GaussianProcessFunction1DWithHyperbolicTangentLengthScaleFunction;
import seed.minerva.ts.HyperbolicTangentLengthScaleFunction1D;
import seed.minerva.ts.ThomsonScattering;
import seed.minerva.util.PrintGraph;

/**
 * This class builds a model with manetics and equilibrium constraint only
 * The diagnostics used are: pf coils, magnetics coils, flux loops
 * Name of graphical model: JetEqui2DModel4
 * 
 * @author lynton
 *                                            
 */
public class ModelBuilder {

	public static void main(String[] args) throws Exception {	
		dump("running in voEqui_magnetics_lynton");
		buildJetEqui2DModel4();
	}
	
	public static GraphicalModel buildJetEqui2DModel4() throws Exception {
		GraphicalModel g = new GraphicalModel("JetEqui2DModel4");
		
		ConstantInt pulse = new ConstantInt(g, "pulse", 81200);
		Constant time = new Constant(g, "time", 52.2280); 
		FirstWallJET firstwall = new FirstWallJET(g, "firstwall", pulse);

		PulseInfo pulseInfo = new PulseInfo(g, "pulseInfo", pulse);

		// EFIT
		GraphicalModel efit = new GraphicalModel(g, "efit");
		JetEfitPpfPsiDataSource efit_ds = new JetEfitPpfPsiDataSource(efit, "efit_ds", pulse, time);	
		//CubicInterpolation2DNode poloidalFluxInterp_efit = new CubicInterpolation2DNode(efit, "poloidalFluxInterp", new Source(efit_ds,  "getR"), new Source(efit_ds,  "getZ"), new Source(efit_ds,  "getFlux"));		
		//EvalScalar2DOnGrid poloidalFluxGridValues_efit = new EvalScalar2DOnGrid(efit, "poloidalFluxGridValues", new Source(efit_ds,  "getR"), new Source(efit_ds,  "getZ"), poloidalFluxInterp_efit);				
		PoloidalFluxGrid poloidalFlux_efit = new PoloidalFluxGrid(efit, "poloidalFlux");		
		poloidalFlux_efit.set("gridFlux", new Source(efit_ds, "getFlux"));
		poloidalFlux_efit.set("gridR", new Source(efit_ds, "getR"));
		poloidalFlux_efit.set("gridZ", new Source(efit_ds, "getZ"));	
		FluxContouringOps psiNOps_efit = new FluxContouringOps(efit, "psiNOps");
		psiNOps_efit.set("poloidalFlux", poloidalFlux_efit);
		psiNOps_efit.set("fieldZeros", poloidalFlux_efit, "getFieldZeroGridCoords");
		psiNOps_efit.set("bounds", poloidalFlux_efit, "getGridRect");
		psiNOps_efit.set("firstwall", firstwall);	
		psiNOps_efit.set("contourNR", 400);
		psiNOps_efit.set("contourNZ", 400);
		psiNOps_efit.set("targetPsiLCFSToFOFS",  1e-10);
		FluxMapWithPrivateRegion fluxPrivReg_efit = new FluxMapWithPrivateRegion(efit, "fluxPrivReg");
		fluxPrivReg_efit.set("normalisedFlux", psiNOps_efit);
		fluxPrivReg_efit.set("xPoints", new Source(psiNOps_efit, "getXPointsInVessel"));
	
		
		// Diagnostic graph
		GraphicalModel diagnostics = new GraphicalModel(g, "diagnostics");
		
		
		// CT / currents
		GraphicalModel currents = new GraphicalModel(g, "currents");
		GraphicalModel pf = new GraphicalModel(currents, "pf");
		GraphicalModel jtor = new GraphicalModel(currents, "jtor");
		GraphicalModel iron = new GraphicalModel(currents, "iron");
		
		JETCoilCurrentsDataSource coilCurrents = new JETCoilCurrentsDataSource(pf, "coilCurrents", pulse, time);
		
		BoundedCellGrid2D plasmaBeamGrid = new BoundedCellGrid2D(jtor, "plasmaBeamGrid", 1.65, 4.05, 30, -1.9, 2.15, 33, new Source(firstwall, "getR"), new Source(firstwall, "getZ"));		
		
		
		/*
		EquilibriumModelOnBeamSet equi_efit = new EquilibriumModelOnBeamSet(efit, "equi");
		
		equi_efit.set("normalisedFlux", psiNOps_efit);
		equi_efit.set("currentBeamR", plasmaBeamGrid, "getx");
		equi_efit.set("currentBeamZ", plasmaBeamGrid, "gety");
		equi_efit.set("currentBeamW", plasmaBeamGrid, "getdx");
		equi_efit.set("currentBeamH", plasmaBeamGrid, "getdy");		
		equi_efit.set("currentBeamJ", new FillArray(efit, "nocurrent", 0.0, new Source(plasmaBeamGrid, "numCells")));
		equi_efit.set("xPoint", psiNOps_efit, "getFirstXPointApprox");	
		Divide scaledPprime = new Divide(efit, "scaledPprime", new Source(efit_ds, "evalPprime"), UnitManager.getInstance().getUnit("pPrime"));
		equi_efit.set("pprime", new Source(scaledPprime, "evalScalar1D"));
		Divide scaledFfprime = new Divide(efit, "scaledFfprime", new Source(efit_ds, "evalFfprime"), UnitManager.getInstance().getUnit("ffPrime"));
		equi_efit.set("ffprime", new Source(scaledFfprime, "evalScalar1D"));
	
		DensityToBeamsAdapter densityToBeams_efit = new DensityToBeamsAdapter(efit, "densityToBeams", new Source(equi_efit, "getEquiCurrentDensity"), new Source(plasmaBeamGrid, "getdx"), new Source(plasmaBeamGrid, "getdy"));				
		MagneticModelAxiSym2 magneticModel_efit = new MagneticModelAxiSym2(efit, "magneticModel");
		magneticModel_efit.set("plasmaBeamCurrents", densityToBeams_efit);
		magneticModel_efit.set("plasmaBeamR", plasmaBeamGrid, "getx");
		magneticModel_efit.set("plasmaBeamZ", plasmaBeamGrid, "gety");
		magneticModel_efit.set("plasmaBeamWidth", plasmaBeamGrid, "getdx");
		magneticModel_efit.set("plasmaBeamHeight", plasmaBeamGrid, "getdy");
		*/
		
		
		
		
		GPSquaredExponentialND plasmaBeamCov = new GPSquaredExponentialND(jtor, "plasmaBeamCov", 606.0, new double[] {0.73, 0.91}, 1.0, new Source(plasmaBeamGrid, "getxy"));
				
		//FillArray plasmaBeamCurrentDensitiesMean = new FillArray(jtor, "plasmaBeamCurrentDensitiesMean", 0.0, new Source(plasmaBeamGrid, "numCells"));
		MultivariateNormal plasmaBeamCurrentDensities = new MultivariateNormal(jtor, "plasmaBeamCurrentDensities", new Source(plasmaBeamCov, "getMean"), plasmaBeamCov);	
		//plasmaBeamCurrentDensities.setFullValue(OneLiners.fillArray(0.0, plasmaBeamCurrentDensities.dim()));
		DensityToBeamsAdapter densityToBeams = new DensityToBeamsAdapter(jtor, "densityToBeams", plasmaBeamCurrentDensities, new Source(plasmaBeamGrid, "getdx"), new Source(plasmaBeamGrid, "getdy"));
				
		MagneticConfigurationSource ironModel = new MagneticConfigurationSource(iron, "ironModel", new MinervaSettingsProperty(iron, "irondef", "jet.ironmodel.default"));		
		
		JetMagneticConfigurationDataSource pfconf_ds = new JetMagneticConfigurationDataSource(pf, "pfconf_ds", pulse);
		//MagneticConfigurationSource pfModel = new MagneticConfigurationSource(pf, "pfModel", "file://c:/data/jet/xbasecache/localXML/jet-pf.xml");
		MagneticConfigurationSource pfModel = new MagneticConfigurationSource(pf, "pfModel", new Source(pfconf_ds, "getXmlId"));

		FillBooleanArray plasmaBeamEnable = new FillBooleanArray(jtor, "plasmaBeamEnable", true, new Source(plasmaBeamGrid, "numCells"));
		MagneticModelAxiSym2 magneticModel = new MagneticModelAxiSym2(currents, "magneticModel");
		magneticModel.set("plasmaBeamCurrents", densityToBeams);
		magneticModel.set("plasmaBeamR", plasmaBeamGrid, "getx");
		magneticModel.set("plasmaBeamZ", plasmaBeamGrid, "gety");
		magneticModel.set("plasmaBeamWidth", plasmaBeamGrid, "getdx");
		magneticModel.set("plasmaBeamHeight", plasmaBeamGrid, "getdy");
		magneticModel.set("plasmaBeamEnable", plasmaBeamEnable);
		
		FillArray ironCurrentsMean = new FillArray(iron, "ironCurrentsMean", 0.0, new Source(ironModel, "numCircuits"));
		FillArray ironCurrentsVar = new FillArray(iron, "ironCurrentsVar", 1e3*1e3/10.0, new Source(ironModel, "numCircuits"));
		MultivariateNormal ironCurrents = new MultivariateNormal(iron, "ironCurrents", ironCurrentsMean, ironCurrentsVar);
		ironCurrents.setFullValue(OneLiners.fillArray(0.0, ironCurrents.dim()));
		magneticModel.set("ironModelId", ironModel);
		magneticModel.set("ironCurrents", ironCurrents);
		
		magneticModel.set("pfModelId", pfModel);
		magneticModel.set("pfCurrents", new Source(coilCurrents, "getPfCurrents"));
		magneticModel.set("totalTFCurrent", new Source(coilCurrents, "getTotalTFCurrent"));
				
		PoloidalFluxGrid poloidalFlux_ct = new PoloidalFluxGrid(currents, "poloidalFlux", new Source(magneticModel, "vectorPotential"), 1.64, 4.06, 40, -1.90, 2.16, 50);
		
		FluxContouringOps psiNOps_ct = new FluxContouringOps(currents, "psiNOps");
		psiNOps_ct.set("poloidalFlux", poloidalFlux_ct);
		psiNOps_ct.set("fieldZeros", poloidalFlux_ct, "getFieldZeroGridCoords");
		psiNOps_ct.set("bounds", poloidalFlux_ct, "getGridRect");
		psiNOps_ct.set("firstwall", firstwall);
		psiNOps_ct.set("contourNR", 400);
		psiNOps_ct.set("contourNZ", 400);
		psiNOps_ct.set("targetPsiLCFSToFOFS",  1e-10);
		
		FluxMapWithPrivateRegion fluxPrivReg_ct = new FluxMapWithPrivateRegion(currents, "fluxPrivReg");
		fluxPrivReg_ct.set("normalisedFlux", psiNOps_ct);
		fluxPrivReg_ct.set("xPoints", new Source(psiNOps_ct, "getXPointsInVessel"));		
		
		QProfile qProfile = new QProfile(g, "qProfile");
		qProfile.set("contourer", psiNOps_ct);
		qProfile.set("psiMagAxis", psiNOps_ct, "getAccurateMagneticAxisPsi");
		qProfile.set("psiLCFS", psiNOps_ct, "getAccuratePsiLCFS");
		qProfile.set("vacuumField", poloidalFlux_ct, "getVacuumFieldAt1m");
				
		
		// Add a constraint node for an optional keeping of the outermost plasma beams to low values
		FillArray boundaryBeamsConstraintValue = new FillArray(jtor, "boundaryBeamsConstraintValue", 0.0, new Source(plasmaBeamGrid, "numCells"));	
		MultivariateNormal boundaryBeamsConstraint = new MultivariateNormal(jtor, "boundaryBeamsConstraint", plasmaBeamCurrentDensities, 1.0*1.0, boundaryBeamsConstraintValue, ProbabilityNode.OBSERVED);
		boundaryBeamsConstraint.setEnable(new Source(plasmaBeamGrid, "isOuterMostCell"));
		boundaryBeamsConstraint.setActive(false);
		
		// Add nodes to get sigmax for the plasma beam GP from one parameter, with sigmaZ=1.5*sigmaR
		Uniform sigmaR = new Uniform(jtor, "sigmaR", 0.05, 2.0, 0.5);
		sigmaR.setActive(false);
		ConstantArray sigmaMul = new ConstantArray(jtor, "sigmaMul", new double[] { 1, 3.0/2.0 }); // sigmaZ = 1.5*sigmaR
		Mul sigmax = new Mul(jtor, "sigmax", sigmaR, sigmaMul);
		plasmaBeamCov.setSigmax(new Source(sigmax, "evalDoubleArray"));
		
		Uniform sigmaf = new Uniform(jtor, "sigmaf", 100.0, 2000.0, 606.0);
		sigmaf.setActive(false);
		plasmaBeamCov.setSigmaf(sigmaf);
		
		CurrentAndEnergyMoments moments = new CurrentAndEnergyMoments(jtor, "moments");
		moments.set("toroidalCurrentDensity", magneticModel, "toroidalCurrentDensity");
		moments.set("normalisedFlux", psiNOps_ct);
		moments.set("vacuumField", poloidalFlux_ct, "getVacuumFieldAt1m");
		moments.set("firstWall", firstwall);
		moments.set("lastClosedFluxSurface", psiNOps_ct, "getLCFS");
		//moments.set("poloidalField", magnetics);
		moments.set("poloidalField", poloidalFlux_ct, "magneticField");
		
		moments.set("pressure", null); // not available unless equi setup later
		moments.set("poloidalCurrentFlux", null);  // not available unless equi setup later
		
		// Add a node for doing different calculations on the toroidal current
		ToroidalCurrentBeamCalcs toroidalCurrentCalcs = new ToroidalCurrentBeamCalcs(jtor, "toroidalCurrentCalcs", new Source(plasmaBeamGrid, "getx"), new Source(plasmaBeamGrid, "gety"), new Source(plasmaBeamGrid, "getdx"), new Source(plasmaBeamGrid, "getdy"), plasmaBeamCurrentDensities, psiNOps_ct, new Source(magneticModel, "magneticField"), qProfile);
				
		FluxCoordinateTransform fluxSource = fluxPrivReg_ct; //fluxPrivReg_efit;
		FluxContouringOps psiNOps = psiNOps_ct; 
		
		// Equilibrium
		GraphicalModel equiG = new GraphicalModel(g, "equi");	

		ConstantInt numProfileValues = new ConstantInt(equiG, "numProfileValues", 20);
		LinSpaceArray npsi = new LinSpaceArray(equiG, "npsi", 0.0, 1.1, numProfileValues);
		FillArray profile_low = new FillArray(equiG, "profile_low", 0.0, numProfileValues);
		
//		// Electron density
//		GraphicalModel neG = new GraphicalModel(equiG, "ne");
//		
//		Uniform ne_l1 = new Uniform(neG, "l1", 0.05, 5.0, 1.0, ProbabilityNode.FREE);
//		Uniform ne_l2 = new Uniform(neG, "l2", 0.01, 0.2, 0.05, ProbabilityNode.FREE);
//		Uniform ne_xw = new Uniform(neG, "xw", 0.02, 0.2, 0.1, ProbabilityNode.FREE);
//		Uniform ne_x0 = new Uniform(neG, "x0", 0.8, 1.0, 0.9, ProbabilityNode.FREE);
//		HyperbolicTangentLengthScaleFunction1D ne_sigmax = new HyperbolicTangentLengthScaleFunction1D(neG, "sigmax", ne_l1, ne_l2, ne_xw, ne_x0);
//		
//		Uniform ne_sigmaf = new Uniform(neG, "sigmaf", 0.1, 20.0, 5.0, ProbabilityNode.FREE);
//		
//		GPHigdonSwallKernGaussianKernel1D ne_prior = new GPHigdonSwallKernGaussianKernel1D(neG, "prior", ne_sigmaf, ne_sigmax, 0.001, npsi);
//		
//		TruncatedMultivariateNormal ne_values = new TruncatedMultivariateNormal(neG, "values", new Source(ne_prior, "getMean"), ne_prior, profile_low, null, null, ProbabilityNode.FREE);
//		Interpolation1DNode ne_1d = new Interpolation1DNode(neG, "ne1d", npsi, new Source(ne_values, "getFullValue"), Interpolation1DNode.INTERPOLATION_MODE_LINEAR, Interpolation1D.EXTRAPOLATE_CONSTANT_VALUE, 0.0);
//		ne_1d.setInterpolationMode(Interpolation1DNode.INTERPOLATION_MODE_CUBIC);
//		FluxMappedFunction ne_3d = new FluxMappedFunction(neG, "ne3d", ne_1d, fluxSource, FluxMappedFunction.NAN_PSIN_RETURN_VAL, 0.0);
//		EvalScalarFunction1D ne_edge_eval = new EvalScalarFunction1D(neG, "edge_eval", ne_1d, new double[] { 1.1 });
//		Normal ne_edge_vobs = new Normal(neG, "ne_edge_vobs", new Source(ne_edge_eval, "evalFirstPos") , 1e-3, 0.0, ProbabilityNode.OBSERVED);
//		ne_values.forceMeansAndSigmasAsTruncatedMarginals(true);
//		
//		// Electron temperature		
//		GraphicalModel teG = new GraphicalModel(equiG, "te");
//		Uniform te_l1 = new Uniform(teG, "l1", 0.05, 5.0, 1.0, ProbabilityNode.FREE);
//		Uniform te_l2 = new Uniform(teG, "l2", 0.01, 0.2, 0.05, ProbabilityNode.FREE);
//		Uniform te_xw = new Uniform(teG, "xw", 0.02, 0.2, 0.1, ProbabilityNode.FREE);
//		Uniform te_x0 = new Uniform(teG, "x0", 0.8, 1.0, 0.9, ProbabilityNode.FREE);
//		HyperbolicTangentLengthScaleFunction1D te_sigmax = new HyperbolicTangentLengthScaleFunction1D(teG, "sigmax", te_l1, te_l2, te_xw, te_x0);		
//		Uniform te_sigmaf = new Uniform(teG, "sigmaf", 0.1, 20.0, 5.0, ProbabilityNode.FREE);		
//		GPHigdonSwallKernGaussianKernel1D te_prior = new GPHigdonSwallKernGaussianKernel1D(teG, "prior", te_sigmaf, te_sigmax, 0.001, npsi);
//			
//		// GPSquaredExponentialND te_prior = new GPSquaredExponentialND(teG, "prior", 20.0, new double[] { 1.0 }, 0.001, npsi);
//		TruncatedMultivariateNormal te_values = new TruncatedMultivariateNormal(teG, "values", new Source(te_prior, "getMean"), te_prior, profile_low, null, null, ProbabilityNode.FREE);
//		Interpolation1DNode te_1d = new Interpolation1DNode(teG, "te1d", npsi, new Source(te_values, "getFullValue"), Interpolation1DNode.INTERPOLATION_MODE_LINEAR, Interpolation1D.EXTRAPOLATE_CONSTANT_VALUE, 0.0);
//		te_1d.setInterpolationMode(Interpolation1DNode.INTERPOLATION_MODE_CUBIC);
//		FluxMappedFunction te_3d = new FluxMappedFunction(teG, "te3d", te_1d, fluxSource, FluxMappedFunction.NAN_PSIN_RETURN_VAL, 0.0);
//		EvalScalarFunction1D te_edge_eval = new EvalScalarFunction1D(teG, "edge_eval", te_1d, new double[] { 1.1 });
//		Normal te_edge_vobs = new Normal(teG, "te_edge_vobs", new Source(te_edge_eval, "evalFirstPos") , 1e-3, 0.0, ProbabilityNode.OBSERVED);
//		te_values.forceMeansAndSigmasAsTruncatedMarginals(true);
//		
//		// Ion pressure
//		GraphicalModel pionG = new GraphicalModel(equiG, "pion");
//		Uniform pion_l1 = new Uniform(pionG, "l1", 0.05, 5.0, 1.0, ProbabilityNode.FREE);
//		Uniform pion_l2 = new Uniform(pionG, "l2", 0.01, 0.2, 0.05, ProbabilityNode.FREE);
//		Uniform pion_xw = new Uniform(pionG, "xw", 0.02, 0.2, 0.1, ProbabilityNode.FREE);
//		Uniform pion_x0 = new Uniform(pionG, "x0", 0.8, 1.0, 0.9, ProbabilityNode.FREE);
//		HyperbolicTangentLengthScaleFunction1D pion_sigmax = new HyperbolicTangentLengthScaleFunction1D(pionG, "sigmax", pion_l1, pion_l2, pion_xw, pion_x0);
//		
//		Uniform pion_sigmaf = new Uniform(pionG, "sigmaf", 0.1, 20.0, 5.0, ProbabilityNode.FREE);		
//		GPHigdonSwallKernGaussianKernel1D pion_prior = new GPHigdonSwallKernGaussianKernel1D(pionG, "prior", pion_sigmaf, pion_sigmax, 0.001, npsi);
//		
//		//GPSquaredExponentialND pion_prior = new GPSquaredExponentialND(pionG, "prior", 20.0, new double[] { 1.0 }, 0.001, npsi);
//		TruncatedMultivariateNormal pion_values = new TruncatedMultivariateNormal(pionG, "values", new Source(pion_prior, "getMean"), pion_prior, profile_low, null, null, ProbabilityNode.FREE);
//		Interpolation1DNode pion_1d = new Interpolation1DNode(pionG, "pion1d", npsi, new Source(pion_values, "getFullValue"), Interpolation1DNode.INTERPOLATION_MODE_LINEAR, Interpolation1D.EXTRAPOLATE_CONSTANT_VALUE, 0.0);
//		pion_1d.setInterpolationMode(Interpolation1DNode.INTERPOLATION_MODE_CUBIC);
//		FluxMappedFunction pion_3d = new FluxMappedFunction(pionG, "pion3d", pion_1d, fluxSource, FluxMappedFunction.NAN_PSIN_RETURN_VAL, 0.0);
//		EvalScalarFunction1D pion_edge_eval = new EvalScalarFunction1D(pionG, "edge_eval", pion_1d, new double[] { 1.1 });
//		Normal pion_edge_vobs = new Normal(pionG, "pion_edge_vobs", new Source(pion_edge_eval, "evalFirstPos") , 1e-3, 0.0, ProbabilityNode.OBSERVED);
//		pion_values.forceMeansAndSigmasAsTruncatedMarginals(true);
//
//		GraphicalModel pressureG = new GraphicalModel(equiG, "pressure");
//		Mul pe_1d_unnorm = new Mul(pressureG, "pe_1d_unnorm", te_1d, ne_1d);
//		Mul pe_1d = new Mul(pressureG, "pe_1d", pe_1d_unnorm, Const.eVnToPascal*UnitManager.get("ElectronDensity")*UnitManager.get("ElectronTemperature")/UnitManager.get("Pressure"));
//		
//		Add pressure_1d = new Add(pressureG, "pressure1d", new Source(pe_1d, "evalScalar1D"), new Source(pion_1d, "evalScalar1D"));
//		
//		
//		/*
//		// Pressure
//		GraphicalModel pressureG = new GraphicalModel(equiG, "pressure");
//		ConstantInt pressure_numValues = new ConstantInt(pressureG, "numValues", 20);
//		LinSpaceArray pressure_npsi = new LinSpaceArray(pressureG, "npsi", 0.0, 1.1, pressure_numValues);
//		GPSquaredExponentialND pressure_prior = new GPSquaredExponentialND(pressureG, "prior", 20.0, new double[] { 1.0 }, 0.001, pressure_npsi);
//		FillArray pressure_low = new FillArray(pressureG, "pressure_low", 0.0, pressure_numValues);
//		TruncatedMultivariateNormal pressure_values = new TruncatedMultivariateNormal(pressureG, "values", new Source(pressure_prior, "getMean"), pressure_prior, pressure_low, null, null, ProbabilityNode.FREE);
//		Interpolation1DNode pressure_1d = new Interpolation1DNode(pressureG, "pressure1d", pressure_npsi, new Source(pressure_values, "getFullValue"), Interpolation1DNode.INTERPOLATION_MODE_LINEAR, Interpolation1D.EXTRAPOLATE_CONSTANT_VALUE, 0.0);
//		FluxMappedFunction pressure_3d = new FluxMappedFunction(pressureG, "pressure3d", pressure_1d, fluxSource, FluxMappedFunction.NAN_PSIN_RETURN_VAL, 0.0);
//		EvalScalarFunction1D pressure_edge_eval = new EvalScalarFunction1D(pressureG, "edge_eval", pressure_1d, new double[] { 1.1 });
//		Normal pressure_edge_vobs = new Normal(pressureG, "edge_vobs", new Source(pressure_edge_eval, "evalFirstPos") , 1e-3, 0.0, ProbabilityNode.OBSERVED);
//		pressure_values.forceMeansAndSigmasAsTruncatedMarginals(true);
//		*/
//		
//		
//		Differentation1D pprime_1d_unnormalised = new Differentation1D(equiG, "pprime1d_unnormalised", pressure_1d);
//		pprime_1d_unnormalised.set("epsilon", 0.005);
//		pprime_1d_unnormalised.set("xgrid", new Source(npsi, "getValue"));
//		Subtract fluxDiff = new Subtract(equiG, "fluxDiff", new Source(psiNOps, "getAccuratePsiLCFS"), new Source(psiNOps, "getAccurateMagneticAxisPsi"));
//		Divide convFactor = new Divide(equiG, "convFactor", 2*Math.PI*UnitManager.get("Pressure")/UnitManager.get("pPrime"), new Source(fluxDiff, "evalConstant")); // Grad Shafranov equation as it stands have pprime and ffprime in units e.g. Pa/(Wb/radian), so actual flux (not normalised) but per radian
//		Mul pprime_1d = new Mul(equiG, "pprime1d", pprime_1d_unnormalised, new Source(convFactor, "evalConstant"));		
		
//  KEEP THIS  for ffprime (LCA)!!!!
		GraphicalModel ffprimeG = new GraphicalModel(equiG, "ffprime");
		Uniform ffprime_l1 = new Uniform(ffprimeG, "l1", 0.05, 5.0, 0.5, ProbabilityNode.FREE);
		Uniform ffprime_l2 = new Uniform(ffprimeG, "l2", 0.01, 0.2, 0.02, ProbabilityNode.FREE);
		Uniform ffprime_xw = new Uniform(ffprimeG, "xw", 0.02, 0.2, 0.1, ProbabilityNode.FREE);
		Uniform ffprime_x0 = new Uniform(ffprimeG, "x0", 0.8, 1.0, 0.9, ProbabilityNode.FREE);
		HyperbolicTangentLengthScaleFunction1D ffprime_sigmax = new HyperbolicTangentLengthScaleFunction1D(ffprimeG, "sigmax", ffprime_l1, ffprime_l2, ffprime_xw, ffprime_x0);		
		Uniform ffprime_sigmaf = new Uniform(ffprimeG, "sigmaf", 0.1, 1e5, 100.0, ProbabilityNode.FREE);		
		GPHigdonSwallKernGaussianKernel1D ffprime_prior = new GPHigdonSwallKernGaussianKernel1D(ffprimeG, "prior", ffprime_sigmaf, ffprime_sigmax, 0.001, npsi);	
		//GPSquaredExponentialND ffprime_prior = new GPSquaredExponentialND(ffprimeG, "prior", 100.0, new double[] { 1.0 }, 0.001, npsi);
		MultivariateNormal ffprime_values = new MultivariateNormal(ffprimeG, "values", new Source(ffprime_prior, "getMean"), ffprime_prior, null, ProbabilityNode.FREE);
		Interpolation1DNode ffprime_1d = new Interpolation1DNode(ffprimeG, "ffprime1d", npsi, new Source(ffprime_values, "getFullValue"), Interpolation1DNode.INTERPOLATION_MODE_LINEAR, Interpolation1D.EXTRAPOLATE_CONSTANT_VALUE, 0.0);
//		ffprime_1d.setInterpolationMode(Interpolation1DNode.INTERPOLATION_MODE_CUBIC);
		FluxMappedFunction ffprime_3d = new FluxMappedFunction(ffprimeG, "ffprime3d", ffprime_1d, fluxSource, FluxMappedFunction.NAN_PSIN_RETURN_VAL, 0.0);	

	//  Replicate the ffprime for pprime, to replace the Te,pion,ne parameters (LCA)!!!!
		GraphicalModel pprimeG = new GraphicalModel(equiG, "pprime");
		Uniform pprime_l1 = new Uniform(pprimeG, "l1", 0.05, 5.0, 0.5, ProbabilityNode.FREE);
		Uniform pprime_l2 = new Uniform(pprimeG, "l2", 0.01, 0.2, 0.02, ProbabilityNode.FREE);
		Uniform pprime_xw = new Uniform(pprimeG, "xw", 0.02, 0.2, 0.1, ProbabilityNode.FREE);
		Uniform pprime_x0 = new Uniform(pprimeG, "x0", 0.8, 1.0, 0.9, ProbabilityNode.FREE);
		HyperbolicTangentLengthScaleFunction1D pprime_sigmax = new HyperbolicTangentLengthScaleFunction1D(pprimeG, "sigmax", pprime_l1, pprime_l2, pprime_xw, pprime_x0);		
		Uniform pprime_sigmaf = new Uniform(pprimeG, "sigmaf", 0.1, 1e5, 100.0, ProbabilityNode.FREE);		
		GPHigdonSwallKernGaussianKernel1D pprime_prior = new GPHigdonSwallKernGaussianKernel1D(pprimeG, "prior", pprime_sigmaf, pprime_sigmax, 0.001, npsi);	
		//GPSquaredExponentialND pprime_prior = new GPSquaredExponentialND(pprimeG, "prior", 100.0, new double[] { 1.0 }, 0.001, npsi);
		MultivariateNormal pprime_values = new MultivariateNormal(pprimeG, "values", new Source(pprime_prior, "getMean"), pprime_prior, null, ProbabilityNode.FREE);
		Interpolation1DNode pprime_1d = new Interpolation1DNode(pprimeG, "pprime1d", npsi, new Source(pprime_values, "getFullValue"), Interpolation1DNode.INTERPOLATION_MODE_LINEAR, Interpolation1D.EXTRAPOLATE_CONSTANT_VALUE, 0.0);
//		pprime_1d.setInterpolationMode(Interpolation1DNode.INTERPOLATION_MODE_CUBIC);
		FluxMappedFunction pprime_3d = new FluxMappedFunction(pprimeG, "pprime3d", pprime_1d, fluxSource, FluxMappedFunction.NAN_PSIN_RETURN_VAL, 0.0);	
		// LYNTON 6/4/2018
		PressureFromDifferential pressure_1d = new PressureFromDifferential(equiG, "p_1d");		
		pressure_1d.set("psiMagAxis", psiNOps, "getAccurateMagneticAxisPsi");
		pressure_1d.set("psiLCFS", psiNOps, "getAccuratePsiLCFS");
		pressure_1d.set("pPrime", pprime_1d); 
		//pressure_1d.set("pPrime", new Source(pprime_1d, "evalScalar1D")); 
		//
		
		//PressureFromDifferential pProfile = new PressureFromDifferential(equiG, "pressure_1d");
		//pProfile.set("psiMagAxis", psiNOps_ct, "getAccurateMagneticAxisPsi");
		//pProfile.set("psiLCFS", psiNOps_ct, "getAccuratePsiLCFS");
		//pProfile.set("pPrime", new Source(pprime_1d, "evalScalar1D")); 
		
		PoloidalCurrentFluxFromDifferential fProfile = new PoloidalCurrentFluxFromDifferential(equiG, "f_1d");		
		fProfile.set("psiMagAxis", psiNOps, "getAccurateMagneticAxisPsi");
		fProfile.set("psiLCFS", psiNOps, "getAccuratePsiLCFS");
		fProfile.set("vacuumField", poloidalFlux_ct, "getVacuumFieldAt1m");
		fProfile.set("ffPrime", ffprime_1d); 


		EquilibriumModelOnBeamSet equi = new EquilibriumModelOnBeamSet(equiG, "equi");
		
		equi.set("normalisedFlux", psiNOps);
		equi.set("currentBeamR", plasmaBeamGrid, "getx");
		equi.set("currentBeamZ", plasmaBeamGrid, "gety");
		equi.set("currentBeamW", plasmaBeamGrid, "getdx");
		equi.set("currentBeamH", plasmaBeamGrid, "getdy");		
		equi.set("currentBeamJ", plasmaBeamCurrentDensities);
		equi.set("xPoint", psiNOps, "getFirstXPointApprox");	
		equi.set("pprime", new Source(pprime_1d, "evalScalar1D")); 
		equi.set("ffprime", ffprime_1d);
		
		MultivariateNormal	equiConstraint = new MultivariateNormal(equiG, "equiConstraint");
		equiConstraint.set(MultivariateNormal.MEAN, equi, "getEquiDiffMean");
		equiConstraint.set(MultivariateNormal.COV, equi, "getEquiDiffSigma2");
		equiConstraint.setObserved(true);
		equiConstraint.set(MultivariateNormal.VALUE, equi, "getEquiObservations");
			
		//now connect the QProfile and Moments calcs to include diamagnetic effects (i.e. use the equi derived Bphi)	
		qProfile.set("RBphi", fProfile);
		qProfile.disconnect("vacuumField"); //disconnect vacuum (scalar) f	
		
		moments.set("poloidalCurrentFlux", fProfile);
		moments.set("pressure", new Source(pressure_1d, "evalScalar1D"));
		
		double equiSigma = 5.001e4; // In [A/m2] I think, since the inner calculation is done in Amperes it seems, and no conversion is done.
		equi.setEvalResolution(3, 3);
		//equi.setCoreEvalPsiNCutoff(0.01);
		equi.setCoreSigmaRaiseMagnitude(0);
		equi.setCoreSigmaRaiseScaleLength(0.1);
		//equi.setPsiNLCFS(1.0);
		equi.setEquiCurrentSigma(equiSigma);
		equi.setSeparatrixCurrentSigma(equiSigma);
		equi.setSOLBaseCurrentSigma(equiSigma);
		equi.setSOLSigmaFalloff(1000);
		equi.setXPointUpperLowerBoundaryZ(0);
		equi.setForceSOLZeroCurrent(false);
		equi.setForcePrivateRegionZeroCurrent(false);
		equi.setDiffAsForce(false);		
		// For changing the GS equation balancing (include gradPsi)
		//m.equi.equi.set("fluxDifferential", m.mag.poloidalFlux); 
		
		
		
		// Diagnostics
		
		GraphicalModel diagnostics_magnetics = new GraphicalModel(diagnostics, "magnetics"); 
		JETMagneticDiagnosticsDataSource mag_ds = new JETMagneticDiagnosticsDataSource(diagnostics_magnetics, "mag_ds", pulse);		
		
		PickupCoils2DDataSourceJET2 pickups_ds = new PickupCoils2DDataSourceJET2(diagnostics_magnetics, "pickups_ds", mag_ds, time);
		PickupCoils2D pickups = new PickupCoils2D(diagnostics_magnetics, "pickups", new Source(pickups_ds, "getR"), new Source(pickups_ds, "getZ"), new Source(pickups_ds, "getPoloidalAngle"), new Source(magneticModel, "magneticField"), null);
		ExtractEnable pickups_enable = new ExtractEnable(diagnostics_magnetics, "pickups_enable", new Source(pickups_ds, "getNames"), new Source(efit_ds, "getMagneticProbeNames"), new Source(efit_ds, "getMagneticProbesEnabled"));
		MultivariateNormal pickups_obs = new MultivariateNormal(diagnostics_magnetics, "pickups_obs", pickups, new Source(pickups_ds, "getVariance"), new Source(pickups_ds, "getData"), ProbabilityNode.OBSERVED);
		//pickups_obs.set(Multivariate.ENABLE, pickups_ds, "getEnable");
		pickups_obs.set(Multivariate.ENABLE, pickups_enable);
		
		FluxLoopsDataSourceJET2 fluxloops_ds = new FluxLoopsDataSourceJET2(diagnostics_magnetics, "fluxloops_ds", mag_ds, time);
		Fluxloops2D fluxloops = new Fluxloops2D(diagnostics_magnetics, "fluxloops", new Source(fluxloops_ds, "getR"), new Source(fluxloops_ds, "getZ"), new Source(magneticModel, "vectorPotential"), null);
		ExtractEnable fluxloops_enable = new ExtractEnable(diagnostics_magnetics, "fluxloops_enable", new Source(fluxloops_ds, "getNames"), new Source(efit_ds, "getFluxLoopNames"), new Source(efit_ds, "getFluxLoopsEnabled"));
		MultivariateNormal fluxloops_obs = new MultivariateNormal(diagnostics_magnetics, "fluxloops_obs", fluxloops, new Source(fluxloops_ds, "getVariance"), new Source(fluxloops_ds, "getData"), ProbabilityNode.OBSERVED);
		//fluxloops_obs.set(Multivariate.ENABLE, fluxloops_ds, "getEnable");
		fluxloops_obs.set(Multivariate.ENABLE, fluxloops_enable);
		
		SaddleCoilsDataSourceJET2 saddles_ds = new SaddleCoilsDataSourceJET2(diagnostics_magnetics, "saddles_ds", mag_ds, time);
		SaddleCoils saddles = new SaddleCoils(diagnostics_magnetics, "saddles");		
		saddles.set("Rinner", saddles_ds, "getRinner");
		saddles.set("Zinner", saddles_ds, "getZinner");
		saddles.set("Router", saddles_ds, "getRouter");
		saddles.set("Zouter", saddles_ds, "getZouter");
		saddles.set("geometryFactor", saddles_ds, "getGeometryFactor");
		saddles.set("toroidalExtent", saddles_ds, "getToroidalExtent");
		saddles.set("A", magneticModel, "vectorPotential");		
		ExtractEnable saddles_enable = new ExtractEnable(diagnostics_magnetics, "saddles_enable", new Source(saddles_ds, "getNames"), new Source(efit_ds, "getSaddleLoopNames"), new Source(efit_ds, "getSaddleLoopsEnabled"));
		MultivariateNormal saddles_obs = new MultivariateNormal(diagnostics_magnetics, "saddles_obs", saddles, new Source(saddles_ds, "getVariance"), new Source(saddles_ds, "getData"), ProbabilityNode.OBSERVED);
		//saddles_obs.set(Multivariate.ENABLE, saddles_ds, "getEnable");
		saddles_obs.set(Multivariate.ENABLE, saddles_enable);
				
		
		
		try {			
			AccessorCompiler.generateAccessorJavaFile(g, "accessors.jet.equi2d", "./src/", true);
		
			// Print out graph
			PrintGraph.mustCreateDotImageFile(g, false, "pdf", null, MinervaSettings.getAppsOutputPath() + "/"+g.getName()+".pdf");
			dump("file written to "+ MinervaSettings.getAppsOutputPath() + "/"+g.getName()+".pdf");
		
		} catch (Exception ex) {
			ex.printStackTrace();			
		}		
		
		System.out.println("Created model "+g.getName());
		return g;
	}
	

	

}
