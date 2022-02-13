

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Mustafa Nural
 *  @version 2.0
 *  @date    Tue Apr 18 14:24:14 EDT 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Lasso Regression (L1 Shrinkage/Regularization)
 */

package scalation
package modeling

import scala.collection.mutable.ArrayBuffer

import scalation.mathstat._
import scalation.optimization.LassoAdmm

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegression` class supports multiple linear regression.  In this case,
 *  'x' is multi-dimensional [1, x_1, ... x_k].  Fit the parameter vector 'b' in
 *  the regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  where 'e' represents the residuals (the part not explained by the model).
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  @param x       the data/input m-by-n matrix
 *  @param y       the response/output m-vector
 *  @param fname_  the feature/variable names
 *  @param hparam  the shrinkage hyper-parameter, lambda (0 => OLS) in the penalty term 'lambda * b dot b'
 */
class LassoRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                       hparam: HyperParameter = LassoRegression.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfm = x.dim2 - 1, df = x.dim - x.dim2):

    private val flaw   = flawf ("LassoRegression")                       // flaw function
    private val debug  = debugf ("LassoRegression", true)                // debug function
    private var lambda = hparam ("lambda").toDouble                      // weight to put on regularization

    modelName = "LassoRegression"

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the value of the shrinkage parameter 'lambda'.
     */
    def lambda_ : Double = lambda

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find an optimal value for the shrinkage parameter 'lambda' using Cross Validation
     *  to minimize 'sse_cv'.  The search starts with the low default value for 'lambda'
     *  doubles it with each iteration, returning the minimum 'lambda' and it corresponding
     *  cross-validated 'sse'.
     */
    def findLambda: (Double, Double) =
        var l      = lambda
        var l_best = l
        var sse    = Double.MaxValue
        for i <- 0 to 20 do
            LassoRegression.hp("lambda") = l
            val rrg   = new LassoRegression (x, y)
            val stats = rrg.crossValidate ()
            val sse2  = stats(QoF.sse.ordinal).mean
            banner (s"LassoRegession with lambda = ${rrg.lambda_} has sse = $sse2")
            if sse2 < sse then
                sse = sse2; l_best = l
            end if
            Fit.showQofStatTable (stats)
            l *= 2
        end for
        (l_best, sse)
    end findLambda

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_1 , ... x_k] + e
     *  regularized by the sum of magnitudes of the coefficients.
     *  @see pdfs.semanticscholar.org/969f/077a3a56105a926a3b0c67077a57f3da3ddf.pdf
     *  @see `scalation.optimization.LassoAdmm`
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        b = LassoAdmm.solve (x_, y_, lambda)                             // Alternating Direction Method of Multipliers

        if b(0).isNaN then flaw ("train", s"parameter b = $b")
        debug ("train", s"LassoAdmm estimates parameter b = $b")
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    override def buildModel (x_cols: MatrixD): LassoRegression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new LassoRegression (x_cols, y, null, hparam)
    end buildModel

end LassoRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegression` companion object provides factory methods for the
 *  `LassoRegression` class.
 */
object LassoRegression:

    /** Base hyper-parameter specification for `LassoRegression`
     */
    val hp  = new HyperParameter; hp += ("lambda", 0.01, 0.01)           // L1 regularization/shrinkage parameter

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `LassoRegression` object from a combined data matrix.
     *  @param xy      the combined data matrix
     *  @param fname   the feature/variable names (use null for default)
     *  @param hparam  the hyper-parameters (use null for default)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): LassoRegression =
        new LassoRegression (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `LassoRegression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (use null for default)
     *  @param hparam  the hyper-parameters (use null for default)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): LassoRegression =
            val xn = normalize (x, (x.mean, x.stdev))
            new LassoRegression (xn, y, fname, hparam)
    end rescale

end LassoRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lassoRegressionTest4` main function tests the `LassoRegression` class using
 *  the AutoMPG dataset.  It illustrates using the `Relation` class for reading the
 *  data from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 Bar and R^2 cv vs. the instance index.
 *  > runMain scalation.modeling.lassoRegressionTest4
 */
@main def lassoRegressionAutoMPG (): Unit =

    // import scalation.database.relation.Relation

    banner ("auto_mpg data")
    // val auto_tab = Relation ("auto_mpg_fixed_cleaned.csv", "auto_mpg", null, -1)
    // auto_tab.show ()

    // val auto_mat = MatrixD.load("auto_mpg_fixed_cleaned.csv")
    // // println(auto_mat)

    // banner ("auto_mpg (x, y) dataset")
    // val (x, y) = auto_tab.toMatrixDD (ArrayBuffer.range (1, 7), 0)
    // println (s"x = $x")
    // println (s"y = $y")

    val auto_mat = MatrixD.load("auto_mpg_fixed_cleaned.csv")
    println(auto_mat)

    val x = auto_mat(?, 0 to 6)
    val y = auto_mat(?, 7)

    println (x)
    println (y)

    banner ("LassoRegression for auto_mpg")
    val mod = new LassoRegression (x, y)                           // create a Lasso regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics
    println (s"best (lambda, sse) = ${mod.findLambda}")
    

    banner ("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Forward Selection - LassoRegression", lines = true)
    println (s"rSq = $rSq")

    banner ("Backward Elimination Test")
    val (cols2, rSq2) = mod.backwardElimAll ()                       // R^2, R^2 Bar, R^2 cv
    val k2 = cols2.size
    val t2 = VectorD.range (1, k2)                                   // instance index
    new PlotM (t2, rSq2.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Backward Elimination - LassoRegression", lines = true)
    println (s"rSq = $rSq2")

    banner ("Stepwise Selection Test")
    val (cols3, rSq3) = mod.stepRegressionAll ()                     // R^2, R^2 Bar, R^2 cv
    val k3 = cols3.size
    val t3 = VectorD.range (1, k3)                                   // instance index
    new PlotM (t3, rSq3.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Stepwise Selection - LassoRegression", lines = true)
    println (s"rSq = $rSq3")

end lassoRegressionAutoMPG



@main def lassoRegression_forestfires (): Unit =

    banner ("forestfires data")
    val auto_mat = MatrixD.load("forestfires_cleaned.csv")
    println(auto_mat)

    val x = auto_mat(?, 1 to 12)
    val y = auto_mat(?, 13)

    println (x)
    println (y)

    banner ("LassoRegression for forestfires")
    val mod = new LassoRegression (x, y)                           // create a Lasso regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics
    // println (s"best (lambda, sse) = ${mod.findLambda}")
    

    // banner ("Forward Selection Test")
    // val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
    // val k = cols.size
    // val t = VectorD.range (1, k)                                   // instance index
    // new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
    //            "R^2 vs n for Forward Selection - LassoRegression", lines = true)
    // println (s"rSq = $rSq")

    // banner ("Backward Elimination Test")
    // val (cols2, rSq2) = mod.backwardElimAll ()                       // R^2, R^2 Bar, R^2 cv
    // val k2 = cols2.size
    // val t2 = VectorD.range (1, k2)                                   // instance index
    // new PlotM (t2, rSq2.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
    //            "R^2 vs n for Backward Elimination - LassoRegression", lines = true)
    // println (s"rSq = $rSq2")

    // banner ("Stepwise Selection Test")
    // val (cols3, rSq3) = mod.stepRegressionAll ()                     // R^2, R^2 Bar, R^2 cv
    // val k3 = cols3.size
    // val t3 = VectorD.range (1, k3)                                   // instance index
    // new PlotM (t3, rSq3.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
    //            "R^2 vs n for Stepwise Selection - LassoRegression", lines = true)
    // println (s"rSq = $rSq3")

end lassoRegression_forestfires

