//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Dec 23 13:54:30 EST 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Model: Symbolic Lasso Regression, including Quadratic and Cubic Lasso Regression
 */

package scalation
package modeling

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegression` object supports symbolic Lasso regression that allows
 *  variables/columns to be raised to various powers, e.g., x^2, x^3, x^.5.
 *  IMPORTANT:  must not include INTERCEPT (column of ones) in initial data matrix),
 *  i.e., DO NOT include a column of ones in x (will cause singularity)
 */
object SymLassoRegression:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `LassoRegression` object from a data matrix and a response vector.
     *  Partial support for "Symbolic Lasso Regression" as matrix x can be raised
     *  to several powers (e.g., x^1 and x^2).  Note, x^1 is automatically included.
     *  Note, Lasso Regression will NOT have an INTERCEPT column.
     *  @param x       the initial data/input m-by-n matrix (before expansion)
     *                     must not include an intercept column of all ones
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (use null for default)
     *  @param powers  the set of powers to raise matrix x to
     *  @param cross   whether to include 2-way cross/interaction terms x_i x_j (defaults to true)
     *  @param cross3  whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param hparam  the hyper-parameters (use LassoRegression.hp for default)
     */
    def apply (x: MatrixD, y: VectorD, fname: Array [String],
               powers: Set [Double], cross: Boolean = true, cross3: Boolean = false,
               hparam: HyperParameter = LassoRegression.hp): LassoRegression =
        var xx     = x                                                    // start with multiple regression for x
        var fname_ = fname

        for p <- powers if p != 1 do
            xx       = xx ++^ x~^p                                        // add power terms x^p
            fname_ ++= fname.map ((n) => s"$n^${p.toInt}")
        end for

        if cross then
           xx       = xx ++^ x.crossAll                                   // add 2-way cross terms x_i x_j
           fname_ ++= SymbolicRegression.crossNames (fname)
        end if

        if cross3 then
           xx       = xx ++^ x.crossAll3                                  // add 3-way cross terms x_i x_j x_k
           fname_ ++= SymbolicRegression.crossNames3 (fname)
        end if

        val mod = new LassoRegression (xx, y, fname_, hparam)
        mod.modelName = "SymLassoRegression" + (if cross then "X" else "") + (if cross3 then "XX" else "")
        mod
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SymLassoRegression` object from a data matrix and a response vector.
     *  This factory function provides data rescaling.
     *  @param x          the data/input m-by-n matrix
     *                        (augment with a first column of ones to include intercept in model)
     *  @param y          the response/output m-vector
     *  @param powers     the set of powers to raise matrix x to
     *  @param cross      whether to include 2-way cross/interaction terms x_i x_j (defaults to true)
     *  @param cross3     whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param fname      the feature/variable names (use null for default)
     *  @param hparam     the hyper-parameters (use Regression.hp for default)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 powers: Set [Double], cross: Boolean = true, cross3: Boolean = false,
                 hparam: HyperParameter = Regression.hp): LassoRegression =
        val xn = normalize (x, (x.mean, x.stdev))
        apply (xn, y, fname, powers, cross, cross3, hparam)
    end rescale

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `LassoRegression` object that uses multiple regression to fit a quadratic
     *  surface to the data.  For example in 2D, the quadratic regression equation is
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_0, x_0^2, x_1, x_1^2] + e
     *  @param x       the initial data/input m-by-n matrix (before quadratic term expansion)
     *                     must not include an intercept column of all ones
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (use null for default)
     *  @param cross   whether to include cross terms x_i * x_j
     *  @param hparam  the hyper-parameters ((use LassoRegression.hp for default)
     */
    def quadratic (x: MatrixD, y: VectorD, fname: Array [String],
                   cross: Boolean = false, hparam: HyperParameter = LassoRegression.hp): LassoRegression =
        val mod = apply (x, y, fname, Set (2), false, cross, hparam)
        mod.modelName = "SymLassoRegression.quadratic" + (if cross then "X" else "")
        mod
    end quadratic

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `LassoRegression` object that uses multiple regression to fit a cubic
     *  surface to the data.  For example in 2D, the cubic regression equation is
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_0, x_0^2, x_0^3,
     *                                                   x_1, x_1^2, x_1^3,
     *                                                   x_0*x_1, x_0^2*x_1, x_0*x_1^2] + e
     *  @param x       the initial data/input m-by-n matrix (before quadratic term expansion)
     *                     must not include an intercept column of all ones
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (use null for default)
     *  @param cross   whether to include 2-way cross/interaction terms x_i x_j (defaults to flase)
     *  @param cross3  whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param hparam  the hyper-parameters ((use LassoRegression.hp for default)
     */
    def cubic (x: MatrixD, y: VectorD, fname: Array [String],
               cross: Boolean = false, cross3: Boolean = false,
               hparam: HyperParameter = LassoRegression.hp): LassoRegression =
        val mod = apply (x, y, fname, Set (2, 3), cross, cross3, hparam)
        mod.modelName = "SymLassoRegression.cubic" + (if cross then "X" else "") + (if cross3 then "X" else "")
        mod
    end cubic

end SymLassoRegression

import Example_AutoMPG._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Symbolic Lasso Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymLassoRegressionTest
 */
@main def SymLassoRegressionTest (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("auto_mpg Symbolic Lasso Regression")
    val mod = SymLassoRegression (x, y, x_fname, Set (-2, -1, 0.5, 2))    // add cross-terms and given powers
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Symbolic Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end SymLassoRegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest2` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Quadratic Lasso Regression" (with cross = false) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymLassoRegressionTest2
 */
@main def SymLassoRegressionTest2 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("auto_mpg Quadratic Lasso Regression")
    val mod = SymLassoRegression.quadratic (x, y, x_fname)                // add x^2 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Quadratic Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end SymLassoRegressionTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest3` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Quadratic X Lasso Regression" (with cross = true) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymLassoRegressionTest3
 */
@main def SymLassoRegressionTest3 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("auto_mpg Quadratic X Lasso Regression")
    val mod = SymLassoRegression.quadratic (x, y, x_fname, true)          // add cross-terms and x^2 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Quadratic X Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end SymLassoRegressionTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest4` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Cubic Lasso Regression" (with cross = false) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymLassoRegressionTest4
 */
@main def SymLassoRegressionTest4 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("auto_mpg Cubic Lasso Regression")
    val mod = SymLassoRegression.cubic (x, y, x_fname)                    // add x^2 and x^3 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Cubic Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end SymLassoRegressionTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest5` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Cubic X Lasso Regression" (with cross = true) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymLassoRegressionTest5
 */
@main def SymLassoRegressionTest5 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("auto_mpg Cubic X Lasso Regression")
    val mod = SymLassoRegression.cubic (x, y, x_fname, true)              // add cross-terms, x^2 and x^3 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Cubic X Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end SymLassoRegressionTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest6` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Cubic XX Lasso Regression" (with cross, cross3 = true) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  WARNING: setting cross3 = true can lead to an explotion of terms.
 *  > runMain scalation.modeling.SymLassoRegressionTest6
 */
@main def SymLassoRegressionTest6 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("auto_mpg Cubic XX Lasso Regression")
    val mod = SymLassoRegression.cubic (x, y, x_fname,                  // add x^2 and x^3 terms
                                        true, true)                     // add cross and cross3 terms
    mod.trainNtest ()()                                                 // train and test the model
    println (mod.summary ())                                            // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                     // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Cubic XX Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end SymLassoRegressionTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest7` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Symbolic Lasso Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  This test case performs data rescaling.
 *  > runMain scalation.modeling.SymLassoRegressionTest7
 */
@main def SymLassoRegressionTest7 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("auto_mpg Symbolic Lasso Regression")
    val mod = SymLassoRegression.rescale (x, y, x_fname,
                                          Set (-2, -1, 0.5, 2))          // add cross-terms and given powers
    mod.trainNtest ()()                                                  // train and test the model
    println (mod.summary ())                                             // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                      // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Symbolic Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end SymLassoRegressionTest7

