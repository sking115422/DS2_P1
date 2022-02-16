
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jan 31 20:59:02 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Ridge Regression (L2 Shrinkage/Regularization)
 *
 *  @see math.stackexchange.com/questions/299481/qr-factorization-for-ridge-regression
 *  Ridge Regression using SVD
 *  @see Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
 *
 *  Since regularization reduces near singularity, Cholesky is used as default
 */

package scalation
package modeling

import scala.math.sqrt
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
//import scalation.minima.GoldenSectionLS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegression` class supports multiple linear ridge regression.
 *  In this case, x is multi-dimensional [x_1, ... x_k].  Ridge regression puts
 *  a penalty on the L2 norm of the parameters b to reduce the chance of them taking
 *  on large values that may lead to less robust models.  Both the input matrix x
 *  and the response vector y are centered (zero mean).  Fit the parameter vector
 *  b in the regression equation
 *      y  =  b dot x + e  =  b_1 * x_1 + ... b_k * x_k + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve for the parameter vector b
 *  using the regularized Normal Equations:
 *      b  =  fac.solve (.)  with regularization  x.t * x + λ * I
 *  Five factorization techniques are provided:
 *      'QR'         // QR Factorization: slower, more stable (default)
 *      'Cholesky'   // Cholesky Factorization: faster, less stable (reasonable choice)
 *      'SVD'        // Singular Value Decomposition: slowest, most robust
 *      'LU'         // LU Factorization: similar, but better than inverse
 *      'Inverse'    // Inverse/Gaussian Elimination, classical textbook technique 
 *  @see statweb.stanford.edu/~tibs/ElemStatLearn/
 *  @param x       the centered data/input m-by-n matrix NOT augmented with a first column of ones
 *  @param y       the centered response/output m-vector
 *  @param fname_  the feature/variable names
 *  @param hparam  the shrinkage hyper-parameter, lambda (0 => OLS) in the penalty term 'lambda * b dot b'
 */
class RidgeRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                       hparam: HyperParameter = RidgeRegression.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfm = x.dim2, df = x.dim - x.dim2 - 1):
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`
         // no intercept => correct Degrees of Freedom (DoF); as lambda get larger, need effective DoF

    private val debug     = debugf ("RidgeRegression", false)            // debug function
    private var lambda    = if hparam("lambda") <= 0.0 then findLambda._1
                            else hparam ("lambda").toDouble
    private val algorithm = hparam("factorization")                      // factorization algorithm

    modelName = "RidgeRegression"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the value of the shrinkage parameter lambda.
     */
    def lambda_ : Double = lambda

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a solver for the Modified Normal Equations using the selected
     *  factorization technique.
     *  @param x_  the data/input matrix
     */
    private def solver (x_ : MatrixD = x): Factorization =
        val xtx  = x_.transpose * x_                                     // pre-compute X.t * X
        val ey   = MatrixD.eye (x_.dim, x_.dim2)                         // identity matrix
        val xtx_ = xtx.copy                                              // copy xtx (X.t * X)
        for i <- xtx_.indices do xtx_(i, i) += lambda                    // add lambda to the diagonal

        algorithm match                                                  // select the factorization technique
        case "Fac_QR"       => val xx = x_ ++ (ey * sqrt (lambda))
                               new Fac_QR (xx)                           // QR Factorization
//      case "Fac_SVD"      => new Fac_SVD (x_)                          // Singular Value Decomposition - FIX
        case "Fac_Cholesky" => new Fac_Cholesky (xtx_)                   // Cholesky Factorization
        case "Fac_LU"       => new Fac_LU (xtx_)                         // LU Factorization
        case _              => new Fac_Inverse (xtx_)                    // Inverse Factorization
        end match
    end solver

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *      yy  =  b dot x + e  =  [b_1, ... b_k] dot [x_1, ... x_k] + e
     *  using the least squares method.
     *  @param x_  the data/input matrix
     *  @param y_  the response/ouput vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        val fac = solver (x_)                                            // create selected factorization technique
        fac.factor ()                                                    // factor the matrix, either X or X.t * X

        b = fac match                                                    // solve for coefficient vector b
            case fac: Fac_QR  => fac.solve (y_ ++ new VectorD (y_.dim))
//          case fac: Fac_SVD => fac.solve (y_)
            case _            => fac.solve (x_.transpose * y_)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                            // make predictions
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find an optimal value for the shrinkage parameter λ using Cross Validation
     *  to minimize sse_cv.  The search starts with the low default value for λ
     *  doubles it with each iteration, returning the minimum λ and it corresponding
     *  cross-validated sse.
     */
    def findLambda: (Double, Double) =
        var l      = lambda                                              // start with a small default value
        var l_best = l
        var sse    = Double.MaxValue
        for i <- 0 to 20 do
            RidgeRegression.hp("lambda") = l
            val rrg = new RidgeRegression (x, y)
            val stats = rrg.crossValidate ()
            val sse2 = stats(QoF.sse.ordinal).mean
            banner (s"RidgeRegession with lambda = ${rrg.lambda_} has sse = $sse2")
            if sse2 < sse then { sse = sse2; l_best = l }
//          debug ("findLambda", showQofStatTable (stats))
            l *= 2
        end for
        (l_best, sse)                                                    // best lambda and its sse_cv
    end findLambda

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find an optimal value for the shrinkage parameter 'λ' using Training to
     *  minimize 'sse'.
     *  FIX - try other QoF measures, e.g., sse_cv
     *  @param xx  the  data/input matrix (full or test)
     *  @param yy  the response/output vector (full or test)
     */
    def findLambda2 (xx: MatrixD = x, yy: VectorD = y): Double =

        def f_sse (λ: Double): Double = 
            lambda = λ
            train (xx, yy)
            e = yy - xx * b
            val sse = e dot e
            if sse.isNaN then throw new ArithmeticException ("sse is NaN")
            debug ("findLambda2", s"for lambda = $λ, sse = $sse")
            sse
        end f_sse

//      val gs = new GoldenSectionLS (f_sse _)
//      gs.search ()
        -0.0
    end findLambda2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of vector y = f(x_, b).  It is overridden for speed.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    override def predict (x_ : MatrixD): VectorD = x_ * b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_     the testing/full data/input matrix
     *  @param fname  the array of feature/variable names
     *  @param b      the parameters/coefficients for the model
     *  @param vifs   the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname, b_ : VectorD = b,
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    override def buildModel (x_cols: MatrixD): RidgeRegression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new RidgeRegression (x_cols, y, null, hparam)
    end buildModel

end RidgeRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegression` companion object defines hyper-parameters and provides
 *  factory functions for the `RidgeRegression` class.
 */
object RidgeRegression:

    /** Base hyper-parameter specification for `RidgeRegression`
     */
    val hp = new HyperParameter;
    hp += ("factorization", "Fac_Cholesky", "Fac_Cholesky")            // factorization algorithm
    hp += ("lambda", 0.01, 0.01)                                       // L2 regularization/shrinkage parameter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Ridge Regression from a combined data matrix.
     *  This function centers the data.
     *  @param xy      the ?centered? data/input m-by-n matrix, NOT augmented with a first column of ones
     *                     and the centered response m-vector (combined)
     *  @param fname   the feature/variable names
     *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term lambda * b dot b
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): RidgeRegression =
        val (x, y) = (xy.not(?, col), xy(?, col)) 
        val mu_x = x.mean                                              // column-wise mean of x
        val mu_y = y.mean                                              // mean of y
        val x_c  = x - mu_x                                            // centered x (column-wise)
        val y_c  = y - mu_y                                            // centered y
        new RidgeRegression (x_c, y_c, fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Ridge Regression from a data matrix and response vector.
     *  This function centers the data.
     *  @param x       the centered data/input m-by-n matrix, NOT augmented with a first column of ones
     *  @param y       the centered response/output vector
     *  @param fname   the feature/variable names
     *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term 'lambda * b dot b'
     */
    def apply (x: MatrixD, y: VectorD, fname: Array [String],
               hparam: HyperParameter): RidgeRegression =
        val hp2 = if hparam == null then hp else hparam 
        val mu_x = x.mean                                              // column-wise mean of x
        val mu_y = y.mean                                              // mean of y
        val x_c  = x - mu_x                                            // centered x (column-wise)
        val y_c  = y - mu_y                                            // centered y
        new RidgeRegression (x_c, y_c, fname, hp2)
    end apply

    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): RidgeRegression = ???

end RidgeRegression

@main def ridgeRegAutoMPG (): Unit =

    banner("Auto MPG Data")
    val auto_mat = MatrixD.load("auto_mpg_fixed_cleaned.csv")
    // println(auto_mat)

    val x = auto_mat(?, 0 to 6)
    val y = auto.mat(?, 7)

    banner("Ridge Regression for Auto MPG")
    val mod = new RidgeRegession(x, y)
    mod.trainNtest()
    println(mod.summary)

    banner("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll()
    val k = cols.size
    val t = VectorD.range(1, k)
    new PlotM(t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
            "R^2 vs n for Forward Selection - Ridge Regression", lines = true)
    println(s"rSq = $rSq")

    banner("Backward Elimination Test")
    val (cols2, rSq2) = mod.backwardElimAll ()                       // R^2, R^2 Bar, R^2 cv
    val k2 = cols2.size
    val t2 = VectorD.range (1, k2)                                   // instance index
    new PlotM (t2, rSq2.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Backward Elimination - Ridge Regression", lines = true)
    println (s"rSq = $rSq2")

    banner("Stepwise Selection Test")
    val (cols3, rSq3) = mod.stepRegressionAll ()                     // R^2, R^2 Bar, R^2 cv
    val k3 = cols3.size
    val t3 = VectorD.range (1, k3)                                   // instance index
    new PlotM (t3, rSq3.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Stepwise Selection - Ridge Regression", lines = true)
    println (s"rSq = $rSq3")

end ridgeRegAutoMPG

@main def ridgeRegForestFirest (): Unit =

    banner("Forest Fires Data")
    val auto_mat = MatrixD.load("forestfires.csv")
    // println(auto_mat)

    val x = auto_mat(?, 0 to 11)
    val y = auto.mat(?, 12)

    banner("Ridge Regression for Forest Fires")
    val mod = new RidgeRegession(x, y)
    mod.trainNtest()
    println(mod.summary)

    banner("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll()
    val k = cols.size
    val t = VectorD.range(1, k)
    new PlotM(t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
            "R^2 vs n for Forward Selection - Ridge Regression", lines = true)
    println(s"rSq = $rSq")

    banner("Backward Elimination Test")
    val (cols2, rSq2) = mod.backwardElimAll ()                       // R^2, R^2 Bar, R^2 cv
    val k2 = cols2.size
    val t2 = VectorD.range (1, k2)                                   // instance index
    new PlotM (t2, rSq2.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Backward Elimination - Ridge Regression", lines = true)
    println (s"rSq = $rSq2")

    banner("Stepwise Selection Test")
    val (cols3, rSq3) = mod.stepRegressionAll ()                     // R^2, R^2 Bar, R^2 cv
    val k3 = cols3.size
    val t3 = VectorD.range (1, k3)                                   // instance index
    new PlotM (t3, rSq3.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Stepwise Selection - Ridge Regression", lines = true)
    println (s"rSq = $rSq3")

end ridgeRegForestFirest

@main def ridgeRegBikeSharingHour (): Unit = 

    banner("Bike Sharing Hour Data")
    val auto_mat = MatrixD.load("bike_sharing_hour.csv")
    // println(auto_mat)

    val x = auto_mat(?, 0 to 11)
    val y = auto.mat(?, 14)

    banner("Ridge Regression for Bike Sharing Hour")
    val mod = new RidgeRegession(x, y)
    mod.trainNtest()
    println(mod.summary)

    banner("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll()
    val k = cols.size
    val t = VectorD.range(1, k)
    new PlotM(t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
            "R^2 vs n for Forward Selection - Ridge Regression", lines = true)
    println(s"rSq = $rSq")

    banner("Backward Elimination Test")
    val (cols2, rSq2) = mod.backwardElimAll ()                       // R^2, R^2 Bar, R^2 cv
    val k2 = cols2.size
    val t2 = VectorD.range (1, k2)                                   // instance index
    new PlotM (t2, rSq2.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Backward Elimination - Ridge Regression", lines = true)
    println (s"rSq = $rSq2")

    banner("Stepwise Selection Test")
    val (cols3, rSq3) = mod.stepRegressionAll ()                     // R^2, R^2 Bar, R^2 cv
    val k3 = cols3.size
    val t3 = VectorD.range (1, k3)                                   // instance index
    new PlotM (t3, rSq3.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Stepwise Selection - Ridge Regression", lines = true)
    println (s"rSq = $rSq3")

end ridgeRegBikeSharingHour

@main def ridgeRegCCPP (): Unit =

    banner("CCPP Data")
    val auto_mat = MatrixD.load("CCPP.csv")
    println(auto_mat)

    val x = auto_mat(?, 0 to 3)
    val y = auto.mat(?, 4)

    banner("Ridge Regression for CCPP")
    val mod = new RidgeRegession(x, y)
    mod.trainNtest()
    println(mod.summary)

    banner("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll()
    val k = cols.size
    val t = VectorD.range(1, k)
    new PlotM(t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
            "R^2 vs n for Forward Selection - Ridge Regression", lines = true)
    println(s"rSq = $rSq")

    banner("Backward Elimination Test")
    val (cols2, rSq2) = mod.backwardElimAll ()                       // R^2, R^2 Bar, R^2 cv
    val k2 = cols2.size
    val t2 = VectorD.range (1, k2)                                   // instance index
    new PlotM (t2, rSq2.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Backward Elimination - Ridge Regression", lines = true)
    println (s"rSq = $rSq2")

    banner("Stepwise Selection Test")
    val (cols3, rSq3) = mod.stepRegressionAll ()                     // R^2, R^2 Bar, R^2 cv
    val k3 = cols3.size
    val t3 = VectorD.range (1, k3)                                   // instance index
    new PlotM (t3, rSq3.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Stepwise Selection - Ridge Regression", lines = true)
    println (s"rSq = $rSq3")

end ridgeRegCCPP

@main def ridgeRegWineQuality (): Unit = 

    banner("Wine Quality Data")
    val auto_mat = MatrixD.load("winequality-white_fixed.csv")
    println(auto_mat)

    val x = auto_mat(?, 0 to 10)
    val y = auto.mat(?, 11)

    banner("Ridge Regression for Wine Quality")
    val mod = new RidgeRegession(x, y)
    mod.trainNtest()
    println(mod.summary)

    banner("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll()
    val k = cols.size
    val t = VectorD.range(1, k)
    new PlotM(t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
            "R^2 vs n for Forward Selection - Ridge Regression", lines = true)
    println(s"rSq = $rSq")

    banner("Backward Elimination Test")
    val (cols2, rSq2) = mod.backwardElimAll ()                       // R^2, R^2 Bar, R^2 cv
    val k2 = cols2.size
    val t2 = VectorD.range (1, k2)                                   // instance index
    new PlotM (t2, rSq2.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Backward Elimination - Ridge Regression", lines = true)
    println (s"rSq = $rSq2")

    banner("Stepwise Selection Test")
    val (cols3, rSq3) = mod.stepRegressionAll ()                     // R^2, R^2 Bar, R^2 cv
    val k3 = cols3.size
    val t3 = VectorD.range (1, k3)                                   // instance index
    new PlotM (t3, rSq3.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Stepwise Selection - Ridge Regression", lines = true)
    println (s"rSq = $rSq3")

end ridgeRegWineQuality