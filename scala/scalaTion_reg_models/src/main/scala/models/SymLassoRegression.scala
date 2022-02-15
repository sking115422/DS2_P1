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

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

// import Example_AutoMPG._

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



object AutoMPG:

    /** the names of the predictor variables; the name of response variable is mpg
     */
    val xr_fname = Array ("cylinders", "displacement", "horsepower", "weight",
                          "acceleration", "modelyear", "origin")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("auto_mpg_fixed_cleaned.csv")

    /** the origin column (6) is categorical
     */
    val oxr = xyr.not(?, 7)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    /** the combined data matrix xy with the origin column (6) removed
     */
    val xy = xyr.not(?, 6)                                             // remove the origin column
//  val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (6)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end AutoMPG



object ForestFiresData:

    /** the names of the predictor variables; the name of response variable is mpg
     */
    val xr_fname = Array ("X", "Y", "month", "day",
                          "FFMC", "DMC", "DC", "ISI",
                          "temp", "RH", "wind", "rain")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("forestfires_cleaned.csv")

    /** the origin column (6) is categorical
     */
    val oxr = xyr.not(?, 12)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    /** the combined data matrix xy with the origin column (6) removed
     */
    //  val xy = xyr.not(?, 6)                                         // remove the origin column
    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (12)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end ForestFiresData



object CCPP_Data:

    /** the names of the predictor variables; the name of response variable is mpg
     */
    val xr_fname = Array ("AT", "V", "AP", "RH")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("CCPP.csv")

    /** the origin column (6) is categorical
     */
    val oxr = xyr.not(?, 4)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    /** the combined data matrix xy with the origin column (6) removed
     */
    //  val xy = xyr.not(?, 6)                                         // remove the origin column
    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (4)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end CCPP_Data



// object WineQuality_Data:

//     /** the names of the predictor variables; the name of response variable is mpg
//      */
//     val xr_fname = Array ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
//                           "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")

//     /** the raw combined data matrix 'xyr'
//      */
//     val xyr = MatrixD.load("winequality-white_fixed.csv")

//     /** the origin column (6) is categorical
//      */
//     val oxr = xyr.not(?, 10)
//     val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

//     /** the combined data matrix xy with the origin column (6) removed
//      */
//     //  val xy = xyr.not(?, 6)                                         // remove the origin column
//     val xy = xyr                                                       // use all columns - may cause multi-collinearity

//     private val n = xy.dim2 - 1                                        // last column in xy

//     val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
//     val _1     = VectorD.one (xy.dim)                                  // vector of all ones
//     val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
//     val ox     = _1 +^: x                                              // prepend a column of all ones to x

//     val x_fname: Array [String] = xr_fname.take (10)
//     val ox_fname: Array [String] = Array ("intercept") ++ x_fname

// end WineQuality_Data



// object BikeSharing_Data:

//     /** the names of the predictor variables; the name of response variable is mpg
//      */
//     val xr_fname = Array ("season", "yr", "mnth", "hr", "holiday", "weekday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "register")

//     /** the raw combined data matrix 'xyr'
//      */
//     val xyr = MatrixD.load("winequality-white_fixed.csv")

//     /** the origin column (6) is categorical
//      */
//     val oxr = xyr.not(?, 10)
//     val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

//     /** the combined data matrix xy with the origin column (6) removed
//      */
//     //  val xy = xyr.not(?, 6)                                         // remove the origin column
//     val xy = xyr                                                       // use all columns - may cause multi-collinearity

//     private val n = xy.dim2 - 1                                        // last column in xy

//     val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
//     val _1     = VectorD.one (xy.dim)                                  // vector of all ones
//     val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
//     val ox     = _1 +^: x                                              // prepend a column of all ones to x

//     val x_fname: Array [String] = xr_fname.take (10)
//     val ox_fname: Array [String] = Array ("intercept") ++ x_fname

// end WineQuality_Data



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Symbolic Lasso Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymLassoRegressionTest
 */
@main def SymLassoRegression_AutoMPG (): Unit =
    
    import AutoMPG._

    // banner ("Variable Names in AutoMPG Dataset")
    // println (s"xr_fname = ${stringOf (xr_fname)}")                     // raw dataset
    // println (s"x_fname  = ${stringOf (x_fname)}")                      // origin column removed
    // println (s"ox_fname = ${stringOf (ox_fname)}")                     // intercept (1's) added

    // println (s"x = $x")
    // println (s"y = $y")
    
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

end SymLassoRegression_AutoMPG



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Symbolic Lasso Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymLassoRegressionTest
 */
@main def SymLassoRegression_ForestFires (): Unit =
    
    import ForestFiresData._

    // banner ("Variable Names in forestfires Dataset")
    // println (s"xr_fname = ${stringOf (xr_fname)}")                     // raw dataset
    // println (s"x_fname  = ${stringOf (x_fname)}")                      // origin column removed
    // println (s"ox_fname = ${stringOf (ox_fname)}")                     // intercept (1's) added

    // println (s"x = $x")
    // println (s"y = $y")

    //   for( a <- 0 to 11){
    //       println(s"var col ${a} = ${x (?, a).variance}")
    //   }

    // println(s"sum = ${x (?, 11).sum}")

    val fname = Array ("X", "Y", "month", "day",
                          "FFMC", "DMC", "DC", "ISI",
                          "temp", "RH", "wind", "rain")
    
    banner ("forestfires Symbolic Lasso Regression")
    val mod = SymLassoRegression (x, y, fname, Set (-2, -1, 0.5, 2))    // add cross-terms and given powers
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

end SymLassoRegression_ForestFires



@main def SymLassoRegression_test (): Unit =
    
    import ForestFiresData._

    // banner ("Variable Names in forestfires Dataset")
    // println (s"xr_fname = ${stringOf (xr_fname)}")                     // raw dataset
    // println (s"x_fname  = ${stringOf (x_fname)}")                      // origin column removed
    // println (s"ox_fname = ${stringOf (ox_fname)}")                     // intercept (1's) added

    // println (s"x = $x")
    // println (s"y = $y")

    //   for( a <- 0 to 11){
    //       println(s"var col ${a} = ${x (?, a).variance}")
    //   }

    // println(s"sum = ${x (?, 11).sum}")

    val f_name = Array ("X", "Y", "month", "day",
                        "FFMC", "DMC", "DC", "ISI",
                        "temp", "RH", "wind", "rain")
    
    banner ("forestfires Symbolic Lasso Regression")
    val arr = Array ("")
    val mod = SymbolicRegression.quadratic (x, y, f_name)    // add cross-terms and given powers
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    // for tech <- Predictor.SelectionTech.values do 
    //     banner (s"Feature Selection Technique: $tech")
    //     val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
    //     val k = cols.size
    //     println (s"k = $k, n = ${x.dim2}")
    //     new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
    //                s"R^2 vs n for Symbolic Lasso Regression with $tech", lines = true)
    //     println (s"$tech: rSq = $rSq")
    // end for

end SymLassoRegression_test




//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegressionTest` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Symbolic Lasso Regression", with powers specified in "Set (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.SymLassoRegressionTest
 */
@main def SymLassoRegression_CCPP (): Unit =
    
    import CCPP_Data._

    banner ("Variable Names in CCPP Dataset")
    println (s"xr_fname = ${stringOf (xr_fname)}")                     // raw dataset
    println (s"x_fname  = ${stringOf (x_fname)}")                      // origin column removed
    println (s"ox_fname = ${stringOf (ox_fname)}")                     // intercept (1's) added

    // println (s"x = $x")
    // println (s"y = $y")
    
    banner ("CCPP Symbolic Lasso Regression")
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

end SymLassoRegression_CCPP



// //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// /** The `SymLassoRegressionTest` main function tests the `SymLassoRegression`
//  *  object using the AutoMPG dataset.  Assumes no missing values.
//  *  It tests custom "Symbolic Lasso Regression", with powers specified in "Set (...)" and
//  *  applies forward selection, backward elimination, or stepwise regression.
//  *  > runMain scalation.modeling.SymLassoRegressionTest
//  */
// @main def SymLassoRegression_WineQuality (): Unit =
    
//     import WineQuality_Data._

//     banner ("Variable Names in winequality-white_fixed")
//     println (s"xr_fname = ${stringOf (xr_fname)}")                     // raw dataset
//     println (s"x_fname  = ${stringOf (x_fname)}")                      // origin column removed
//     println (s"ox_fname = ${stringOf (ox_fname)}")                     // intercept (1's) added

//     // println (s"x = $x")
//     // println (s"y = $y")
    
//     banner ("Wine Quality Symbolic Lasso Regression")
//     val mod = SymLassoRegression (x, y, x_fname, Set (-2, -1, 0.5, 2))    // add cross-terms and given powers
//     mod.trainNtest ()()                                                   // train and test the model
//     println (mod.summary ())                                              // parameter/coefficient statistics

//     for tech <- Predictor.SelectionTech.values do 
//         banner (s"Feature Selection Technique: $tech")
//         val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
//         val k = cols.size
//         println (s"k = $k, n = ${x.dim2}")
//         new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
//                    s"R^2 vs n for Symbolic Lasso Regression with $tech", lines = true)
//         println (s"$tech: rSq = $rSq")
//     end for

// end SymLassoRegression_WineQuality